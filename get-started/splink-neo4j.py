import re
import unicodedata
from typing import List, Any, Dict
from pathlib import Path
from datetime import datetime
import time
import logging
import warnings
import sys

import pandas as pd
import duckdb

# --- SPLINK IMPORTS (v4 style) ---
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl
import splink.comparison_level_library as cll

# --- FASTEMBED ---
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------------------
# 0. Output directory and logging / timing setup
# -------------------------------------------------------------------
SCRIPT_START = time.perf_counter()

timestamp = datetime.now().strftime("%H-%M")

OUTPUT_DIR = Path("output")
RUN_DIR = OUTPUT_DIR / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MD_PATH = RUN_DIR / f"output_{timestamp}.md"
TIMING_PATH = RUN_DIR / f"timings_{timestamp}.txt"
LOG_PATH = RUN_DIR / f"log_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def _log_warning(message, category, filename, lineno, file=None, line=None):
    logger.warning("%s in %s:%s: %s", category.__name__, filename, lineno, message)

warnings.showwarning = _log_warning

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
timings: Dict[str, float] = {}


# -------------------------------------------------------------------
# 1. Load your entities
# -------------------------------------------------------------------
logger.info("Loading input_data.json")

t_load0 = time.perf_counter()
df = pd.read_json("input_data_large.json")
df = df.reset_index(drop=True)
df["unique_id"] = df.index.astype("int64")
t_load1 = time.perf_counter()
timings["load_entities"] = t_load1 - t_load0
logger.info("Loaded entities in %.3f seconds", timings["load_entities"])


# -------------------------------------------------------------------
# 2. String feature engineering
# -------------------------------------------------------------------
def normalise_text(value: Any) -> str:
    """Lowercase, remove accents, strip punctuation -> stable token string."""
    if not isinstance(value, str):
        return ""
    v = value.lower()
    v = unicodedata.normalize("NFKD", v)
    v = "".join(c for c in v if not unicodedata.combining(c))
    v = re.sub(r"[^a-z0-9\s]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v

def remove_spaces(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace(" ", "")

def sort_tokens_alpha(value: str) -> str:
    if not isinstance(value, str):
        return ""
    tokens = value.split()
    tokens.sort()
    return " ".join(tokens)

# UPDATED: Handle single tokens correctly for strict acronym matching
def initials_abbrev(value: str) -> str:
    """
    If single token (e.g. 'UAE'), return as is ('uae').
    If multi token (e.g. 'United Arab Emirates'), return first chars ('uae').
    """
    if not isinstance(value, str):
        return ""
    tokens = value.split()
    if not tokens:
        return ""
    # If it's already a single word/acronym, treat the whole word as the 'initials'
    # This allows us to compare norm == initials
    if len(tokens) == 1:
        return tokens[0].lower()
    
    return "".join(t[0].lower() for t in tokens)

t_feat0 = time.perf_counter()

df["entity_id_norm"] = df["entity_id"].apply(normalise_text)
df["entity_id_nospace"] = df["entity_id_norm"].apply(remove_spaces)
df["entity_id_tokens_sorted"] = df["entity_id_norm"].apply(sort_tokens_alpha)
df["entity_id_initials"] = df["entity_id_norm"].apply(initials_abbrev)

t_feat1 = time.perf_counter()
timings["string_feature_engineering"] = t_feat1 - t_feat0
logger.info("String feature engineering took %.3f seconds", timings["string_feature_engineering"])


# -------------------------------------------------------------------
# 3. Description embeddings
# -------------------------------------------------------------------
USE_GPU = True 

t_emb0 = time.perf_counter()

if USE_GPU:
    logger.info("Using GPU-accelerated SentenceTransformer...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target device: {device}")

    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name, device=device)

    desc_texts = [
        "passage: " + (d if d else "") 
        for d in df["description"].fillna("").astype(str).tolist()
    ]

    logger.info("Encoding descriptions...")
    embeddings_numpy = model.encode(
        desc_texts,
        normalize_embeddings=True, 
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    desc_embs = embeddings_numpy.tolist()
    df["description_embedding"] = desc_embs

else:
    logger.info("Using CPU-based FastEmbed...")
    TextEmbedding.add_custom_model(
        model="intfloat/multilingual-e5-small",
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf="intfloat/multilingual-e5-small"),
        dim=384,
        model_file="onnx/model_qint8_avx512_vnni.onnx"
    )

    embedding_model = TextEmbedding(
      model_name="intfloat/multilingual-e5-small",
    )

    desc_texts: List[str] = [
        "passage: " + (d or "") for d in df["description"].fillna("").astype(str).tolist()
    ]

    embeddings_gen = embedding_model.embed(desc_texts)
    desc_embs = [list(e) for e in embeddings_gen]
    
    df["description_embedding"] = pd.Series(desc_embs, index=df.index)

t_emb1 = time.perf_counter()
timings["embedding_generation"] = t_emb1 - t_emb0
logger.info("Embedding generation took %.3f seconds", timings["embedding_generation"])


# -------------------------------------------------------------------
# 4. Splink settings - CONSOLIDATED LOGIC
# -------------------------------------------------------------------
db_api = DuckDBAPI()

# A separate, backup comparison just for embeddings (for cases like "ADGM" vs "Abu Dhabi Global Market")
# This is kept strict to avoid false positives.
embedding_only_comparison = cl.CustomComparison(
    output_column_name="description_embedding",
    comparison_description="Similarity on description embeddings only",
    comparison_levels=[
        cll.NullLevel("description_embedding"),
        cll.CustomLevel(
            "list_cosine_similarity(description_embedding_l, description_embedding_r) >= 0.92",
            label_for_charts="Very High Embedding Similarity"
        ),
        cll.ElseLevel(),
    ],
)

# THE MASTER COMPARISON
# This replaces ALL separate name/initials comparisons.
# It enforces the "Acronym Trap" by structure: if both are acronyms and fail Level 1, 
# they fall to 'Else' (Negative Weight) and never get a chance to be matched by fuzzy rules in Level 2 or 3.
master_identifier_comparison = cl.CustomComparison(
    output_column_name="master_identifier",
    comparison_description="Consolidated Identifier Logic with Acronym Trap",
    comparison_levels=[
        cll.NullLevel("entity_id_norm"),
        
        # LEVEL 1: STRICT ACRONYM MATCH
        # Condition: Both are acronyms. 
        # Requirement: EXACT match + High Embedding.
        cll.CustomLevel(
            """
            (entity_id_norm_l = entity_id_initials_l AND entity_id_norm_r = entity_id_initials_r)
            AND entity_id_initials_l = entity_id_initials_r
            AND list_cosine_similarity(description_embedding_l, description_embedding_r) >= 0.85
            """,
            label_for_charts="Strict Acronym Match (Exact + High Emb)"
        ),

        # LEVEL 2: STANDARD FUZZY NAME MATCH
        # GATED: Only runs if NOT both are acronyms.
        cll.CustomLevel(
            """
            NOT (entity_id_norm_l = entity_id_initials_l AND entity_id_norm_r = entity_id_initials_r)
            AND (
                jaro_winkler_similarity(entity_id_norm_l, entity_id_norm_r) > 0.97
                OR jaro_winkler_similarity(entity_id_nospace_l, entity_id_nospace_r) > 0.97
                OR jaro_winkler_similarity(entity_id_tokens_sorted_l, entity_id_tokens_sorted_r) > 0.97
            )
            AND list_cosine_similarity(description_embedding_l, description_embedding_r) >= 0.80
            """,
            label_for_charts="Standard Fuzzy Name Match (+ Emb)"
        ),

        # LEVEL 3: FUZZY INITIALS MATCH
        # GATED: Only runs if NOT both are acronyms.
        cll.CustomLevel(
            """
            NOT (entity_id_norm_l = entity_id_initials_l AND entity_id_norm_r = entity_id_initials_r)
            AND jaro_winkler_similarity(entity_id_initials_l, entity_id_initials_r) > 0.95
            AND list_cosine_similarity(description_embedding_l, description_embedding_r) >= 0.85
            """,
            label_for_charts="Fuzzy Initials Match (+ Emb)"
        ),
        
        # LEVEL 4: FALLTHROUGH (The Trap's Dungeon)
        # Any pair where (Both are Acronyms AND Mismatch) falls here.
        # Any pair where (Names don't match AND Initials don't match) falls here.
        cll.ElseLevel()
    ]
)

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        # 1. The Master Identifier Comparison (Consolidated)
        master_identifier_comparison,
        
        # 2. Embedding Backup (Strict)
        embedding_only_comparison,
    ],
    blocking_rules_to_generate_predictions=[
        block_on("entity_type"),
    ],
    retain_intermediate_calculation_columns=True,
    em_convergence=0.01,
)


# -------------------------------------------------------------------
# 5. Initialise linker and TRAIN
# -------------------------------------------------------------------
t_linker0 = time.perf_counter()
linker = Linker(df, settings, db_api=db_api)
t_linker1 = time.perf_counter()
timings["linker_init"] = t_linker1 - t_linker0
logger.info("Linker initialisation took %.3f seconds", timings["linker_init"])

# ----------------------------------------------------------------
# DETERMINISTIC RULES (Must match the levels in comparisons)
# ----------------------------------------------------------------
deterministic_rules = ["""
    l.entity_type = r.entity_type
    AND (
        -- CASE A: BOTH ARE ACRONYMS
        (
            (l.entity_id_norm = l.entity_id_initials AND r.entity_id_norm = r.entity_id_initials)
            AND l.entity_id_initials = r.entity_id_initials
            AND list_cosine_similarity(l.description_embedding, r.description_embedding) >= 0.85
        )
        OR
        -- CASE B: AT LEAST ONE IS NOT AN ACRONYM
        (
            NOT (l.entity_id_norm = l.entity_id_initials AND r.entity_id_norm = r.entity_id_initials)
            AND (
                -- Strong Name Match
                (
                    (jaro_winkler_similarity(l.entity_id_norm, r.entity_id_norm) > 0.97
                     OR jaro_winkler_similarity(l.entity_id_nospace, r.entity_id_nospace) > 0.97
                     OR jaro_winkler_similarity(l.entity_id_tokens_sorted, r.entity_id_tokens_sorted) > 0.97)
                    AND list_cosine_similarity(l.description_embedding, r.description_embedding) >= 0.80
                )
                OR
                -- Fuzzy Initials Match
                (
                    jaro_winkler_similarity(l.entity_id_initials, r.entity_id_initials) > 0.95
                    AND list_cosine_similarity(l.description_embedding, r.description_embedding) >= 0.85
                )
            )
        )
        OR
        -- CASE C: EMBEDDING ONLY (Backup)
        list_cosine_similarity(l.description_embedding, r.description_embedding) >= 0.92
    )
"""]

t_pmatch0 = time.perf_counter()
linker.training.estimate_probability_two_random_records_match(
    deterministic_rules,
    recall=0.7, 
)
t_pmatch1 = time.perf_counter()
timings["estimate_probability_two_random_records_match"] = t_pmatch1 - t_pmatch0
logger.info(
    "estimate_probability_two_random_records_match took %.3f seconds",
    timings["estimate_probability_two_random_records_match"],
)

t_u0 = time.perf_counter()
linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
t_u1 = time.perf_counter()
timings["estimate_u_using_random_sampling"] = t_u1 - t_u0
logger.info(
    "estimate_u_using_random_sampling took %.3f seconds",
    timings["estimate_u_using_random_sampling"],
)

# EM training
training_blocking_rule = "l.entity_type = r.entity_type"

t_em0 = time.perf_counter()
linker.training.estimate_parameters_using_expectation_maximisation(
    training_blocking_rule
)
t_em1 = time.perf_counter()
timings["em_training"] = t_em1 - t_em0
logger.info("EM training took %.3f seconds", timings["em_training"])


# -------------------------------------------------------------------
# 6. Predict + cluster
# -------------------------------------------------------------------
output_lines: List[str] = [] 

t_inf0 = time.perf_counter()
pairwise_predictions = linker.inference.predict(
    threshold_match_weight=0.0
)
t_inf1 = time.perf_counter()
timings["inference_predict"] = t_inf1 - t_inf0
logger.info("inference.predict took %.3f seconds", timings["inference_predict"])

t_df0 = time.perf_counter()
preds = pairwise_predictions.as_pandas_dataframe()
t_df1 = time.perf_counter()
timings["pairwise_as_dataframe"] = t_df1 - t_df0

# Columns to show for debugging
cols = [
    "match_weight",
    "match_probability",
    "bf_description_embedding", 
]
for col in ["entity_type_l", "entity_id_l", "entity_type_r", "entity_id_r"]:
    if col in preds.columns:
        cols.insert(0, col)

if not preds.empty:
    top10 = preds.sort_values("match_weight", ascending=False).head(10)[cols]
    print("\n--- Top 10 Pairwise Predictions (markdown) ---")
    top10_md = top10.to_markdown(index=False)
    print(top10_md)
    output_lines.append("\n## Top 10 pairwise predictions\n")
    output_lines.append(top10_md)
else:
    output_lines.append("\n## No predictions found above threshold\n")

# Clustering
t_cluster0 = time.perf_counter()
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions,
    threshold_match_probability=0.80, 
)
t_cluster1 = time.perf_counter()
timings["clustering"] = t_cluster1 - t_cluster0
logger.info(
    "cluster_pairwise_predictions_at_threshold took %.3f seconds",
    timings["clustering"],
)

t_clusters_df0 = time.perf_counter()
df_clusters = clusters.as_pandas_dataframe()
t_clusters_df1 = time.perf_counter()

# -------------------------------------------------------------------
# 7. Build merge plan
# -------------------------------------------------------------------
t_merge0 = time.perf_counter()
merge_plans = []
for cluster_id, g in df_clusters.groupby("cluster_id"):
    if len(g) == 1:
        continue

    canonical = (
        g["entity_id"]
        .astype(str)
        .sort_values(key=lambda s: s.str.len(), ascending=False)
        .iloc[0]
    )

    merge_plans.append(
        {
            "cluster_id": int(cluster_id),
            "merged_entity_name": canonical,
            "unique_ids": g["unique_id"].tolist(),
            "entity_ids": g["entity_id"].tolist(),
            "entity_types": g["entity_type"].unique().tolist(),
        }
    )

merge_df = pd.DataFrame(merge_plans)
t_merge1 = time.perf_counter()
timings["merge_planning"] = t_merge1 - t_merge0
logger.info("Merge planning took %.3f seconds", timings["merge_planning"])

print("\nSuggested merges (markdown):")
if not merge_df.empty:
    merge_md = merge_df.to_markdown(index=False)
    print(merge_md)
    output_lines.append("\n## Suggested merges\n")
    output_lines.append(merge_md)
else:
    print("No merges suggested.")
    output_lines.append("\nNo merges suggested.\n")


# -------------------------------------------------------------------
# 8. Save results
# -------------------------------------------------------------------
SCRIPT_END = time.perf_counter()
timings["total_runtime"] = SCRIPT_END - SCRIPT_START
logger.info("Total script runtime: %.3f seconds", timings["total_runtime"])

output_lines.append("\n## Timing summary (seconds)\n")
for name, sec in timings.items():
    output_lines.append(f"- **{name}**: {sec:.3f}")

with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
    f.write("# Splink entity resolution debug output\n\n")
    f.write("\n".join(output_lines))

print(f"\nAll markdown output written to {OUTPUT_MD_PATH}")