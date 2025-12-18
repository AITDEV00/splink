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
# 1. Load entities and Filter
# -------------------------------------------------------------------
logger.info("Loading input_data.json")

t_load0 = time.perf_counter()
df = pd.read_json("input_data_large.json")

# FILTER STEP: Discard 'data' types immediately
initial_count = len(df)
df = df[~df['entity_type'].isin(['data', 'year', 'date', 'period', 'timeperiod'])].copy()
filtered_count = len(df)
logger.info(f"Filtered out {initial_count - filtered_count} records with entity_type='data', 'year', 'date', 'period', 'timeperiod'")

df = df.reset_index(drop=True)
df["unique_id"] = df.index.astype("int64")
t_load1 = time.perf_counter()
timings["load_entities"] = t_load1 - t_load0
logger.info("Loaded entities in %.3f seconds", timings["load_entities"])


# -------------------------------------------------------------------
# 2. String feature engineering (Hybrid Acronym)
# -------------------------------------------------------------------
def generate_hybrid_acronym(text: Any) -> str:
    """
    Generates an acronym based on Capitals OR Start of Words.
    Example: "Abu Dhabi" -> "AD"
    Example: "Ministry of Finance" -> "MOF"
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Clean special chars, replace with space to preserve word boundaries
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    acronym = []
    is_start_of_word = True
    
    for char in clean_text:
        if char.isspace():
            is_start_of_word = True
            continue
        # Rule: Include if Capital OR Start of Word
        if char.isupper() or is_start_of_word:
            acronym.append(char.upper())
        is_start_of_word = False
        
    return "".join(acronym).lower() # FIXED: Return lowercase to match normalization

# Standard normalization for baseline comparison if needed, 
# though main logic uses the acronym column now.
def normalise_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    v = value.lower()
    v = unicodedata.normalize("NFKD", v)
    v = "".join(c for c in v if not unicodedata.combining(c))
    v = re.sub(r"[^a-z0-9\s]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v

t_feat0 = time.perf_counter()

# Generate the hybrid acronym
df["acronym"] = df["entity_id"].apply(generate_hybrid_acronym)
# Keep standard normalization for backup/display
df["entity_id_norm"] = df["entity_id"].apply(normalise_text)

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
# 4. Splink settings - NEW PLAN LOGIC
# -------------------------------------------------------------------
db_api = DuckDBAPI()

# The specific comparison logic from the plan
# Using list_cosine_similarity to match existing codebase syntax
strategy_comparison = cl.CustomComparison(
    output_column_name="entity_id",
    comparison_description="Hybrid Strategy: Safe Long-Form vs Verified Acronyms",
    comparison_levels=[
        cll.NullLevel("entity_id"),
        
        # Condition A: Safe Long-Form Match (Level 1)
        # Logic: Both IDs must be significantly different from their acronyms (implying they are long forms)
        # AND they must fuzzy match each other.
        # FIXED: Added lower() calls to ensure case-insensitive matching in SQL
        # FIXED: Added length > 4 check to force short codes to Condition B
        # FIXED: Tightened fuzzy threshold to 0.96
        cll.CustomLevel(
            sql_condition="""
                (length(entity_id_norm_l) > 4) AND
                (length(entity_id_norm_r) > 4) AND
                (jaro_winkler_similarity(entity_id_norm_l, lower(acronym_l)) < 0.85) AND 
                (jaro_winkler_similarity(entity_id_norm_r, lower(acronym_r)) < 0.85) AND 
                (jaro_winkler_similarity(entity_id_norm_l, entity_id_norm_r) > 0.96)
            """,
            label_for_charts="Condition A: Safe Long-Form Match"
        ),

        # Condition B: Abbreviation/Acronym Match + Semantic Verification (Level 2)
        # Logic: Acronyms match fuzzily AND Embeddings match strictly.
        # This catches "AD" vs "Abu Dhabi" (where acronyms match) or "AD" vs "AD".
        cll.CustomLevel(
            sql_condition="""
                (jaro_winkler_similarity(lower(acronym_l), lower(acronym_r)) > 0.85) AND 
                (list_cosine_similarity(description_embedding_l, description_embedding_r) > 0.96)
            """,
            label_for_charts="Condition B: Verified Acronym Match"
        ),

        cll.ElseLevel() # Level 3
    ]
)

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        strategy_comparison,
        cl.ExactMatch("entity_type") # Explicit match on entity type
    ],
    blocking_rules_to_generate_predictions=[
        block_on("entity_type"), # Restriction Step: Block on entity_type
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

# Deterministic Rules for Training
# We mirror the levels logic for Probability estimation
# FIXED: Updated rules to use normalized columns, length checks, lower(), and new thresholds
deterministic_rules = ["""
    l.entity_type = r.entity_type
    AND (
        -- Condition A: Safe Long Form
        (
            (length(l.entity_id_norm) > 4) AND
            (length(r.entity_id_norm) > 4) AND
            (jaro_winkler_similarity(l.entity_id_norm, lower(l.acronym)) < 0.85) AND 
            (jaro_winkler_similarity(r.entity_id_norm, lower(r.acronym)) < 0.85) AND 
            (jaro_winkler_similarity(l.entity_id_norm, r.entity_id_norm) > 0.96)
        )
        OR
        -- Condition B: Verified Acronym
        (
            (jaro_winkler_similarity(lower(l.acronym), lower(r.acronym)) > 0.85) AND 
            (list_cosine_similarity(l.description_embedding, r.description_embedding) > 0.96)
        )
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

# --- ADD REASON COLUMN ---
def get_match_reason(gamma_val):
    if gamma_val == 1:
        return "Condition A: Safe Long-Form"
    elif gamma_val == 2:
        return "Condition B: Verified Acronym"
    elif gamma_val == 3:
        return "No Match (Else)"
    return f"Level {gamma_val}"

# Splink typically names the gamma column as 'gamma_{output_column_name}'
if "gamma_entity_id" in preds.columns:
    preds["match_reason"] = preds["gamma_entity_id"].apply(get_match_reason)
else:
    preds["match_reason"] = "Unknown (Gamma col missing)"

# Columns to show for debugging
cols = [
    "match_weight",
    "match_probability",
    "match_reason", # Added the reason
]
for col in ["entity_type_l", "entity_id_l", "acronym_l", "entity_id_r", "acronym_r"]:
    if col in preds.columns:
        cols.insert(0, col)

if not preds.empty:
    # Filter for all matches > 0.5 probability
    high_prob_preds = preds[preds["match_probability"] > 0.5].sort_values("match_weight", ascending=True)
    
    # We display up to 50 to avoid creating massive markdown files, 
    # but the filter logic captures what you asked for.
    display_limit = 50 
    
    print(f"\n--- High Probability Predictions (>0.5) [Top {display_limit} shown] ---")
    
    if len(high_prob_preds) > 0:
        top_md = high_prob_preds.head(display_limit)[cols].to_markdown(index=False)
        print(top_md)
        output_lines.append(f"\n## High Probability Predictions (>0.5) [Top {display_limit} shown]\n")
        output_lines.append(top_md)
    else:
        print("No predictions found with probability > 0.5")
        output_lines.append("\n## No predictions found with probability > 0.5\n")
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

# --- NEW: Pre-calculate match reasons per cluster ---
# 1. Map unique_id to cluster_id
cluster_map = df_clusters.set_index("unique_id")["cluster_id"].to_dict()

# 2. Add cluster IDs to predictions (only if preds not empty)
if not preds.empty and "unique_id_l" in preds.columns and "unique_id_r" in preds.columns:
    # We use map to attach the cluster ID to the left and right sides of the prediction
    preds["cluster_id_l"] = preds["unique_id_l"].map(cluster_map)
    preds["cluster_id_r"] = preds["unique_id_r"].map(cluster_map)
    
    # 3. Filter for edges that are WITHIN the same cluster
    intra_cluster_preds = preds[preds["cluster_id_l"] == preds["cluster_id_r"]].copy()
    
    # 4. Group by cluster ID and collect unique match reasons
    # Result: {101: "Condition A", 102: "Condition A | Condition B"}
    reasons_by_cluster = (
        intra_cluster_preds.groupby("cluster_id_l")["match_reason"]
        .apply(lambda x: " | ".join(sorted(set(x))))
        .to_dict()
    )
else:
    reasons_by_cluster = {}

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

    # Fetch the pre-calculated reason for this cluster
    reason_str = reasons_by_cluster.get(cluster_id, "Indirect Link / Unknown")

    merge_plans.append(
        {
            "cluster_id": int(cluster_id),
            "merged_entity_name": canonical,
            "match_reasons": reason_str,  # Added to output
            "unique_ids": g["unique_id"].tolist(),
            "entity_ids": g["entity_id"].tolist(),
            "acronyms": g["acronym"].tolist(),
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