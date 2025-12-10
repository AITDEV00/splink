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

# --- SPLINK IMPORTS (v4 style) ---
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl
import splink.comparison_level_library as cll

# --- FASTEMBED (intfloat/multilingual-e5-small ONNX) ---
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
import duckdb  # for debug cosine similarity


# -------------------------------------------------------------------
# 0. Output directory and logging / timing setup
# -------------------------------------------------------------------
SCRIPT_START = time.perf_counter()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# hour-minute stamp, e.g. 17-43
timestamp = datetime.now().strftime("%H-%M")

OUTPUT_MD_PATH = OUTPUT_DIR / f"output_{timestamp}.md"
TIMING_PATH = OUTPUT_DIR / f"timings_{timestamp}.txt"
LOG_PATH = OUTPUT_DIR / f"log_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),  # still see logs in console
    ],
)
logger = logging.getLogger(__name__)

# Route Python warnings into the logger
def _log_warning(message, category, filename, lineno, file=None, line=None):
    logger.warning("%s in %s:%s: %s", category.__name__, filename, lineno, message)

warnings.showwarning = _log_warning

# Route uncaught exceptions into the logger
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Let KeyboardInterrupt go through as normal
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# Collect per-step timings here
timings: Dict[str, float] = {}


# -------------------------------------------------------------------
# 1. Load your entities
# -------------------------------------------------------------------
logger.info("Loading input_data.json")

t_load0 = time.perf_counter()
df = pd.read_json("input_data.json")
df = df.reset_index(drop=True)
df["unique_id"] = df.index.astype("int64")
t_load1 = time.perf_counter()
timings["load_entities"] = t_load1 - t_load0
logger.info("Loaded entities in %.3f seconds", timings["load_entities"])


# -------------------------------------------------------------------
# 2. String feature engineering on entity_id
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


def initials_abbrev(value: str) -> str:
    """First char of each token, used as rough abbreviation (ADIB)."""
    if not isinstance(value, str):
        return ""
    tokens = value.split()
    return "".join(t[0] for t in tokens if t)


t_feat0 = time.perf_counter()

df["entity_id_norm"] = df["entity_id"].apply(normalise_text)
df["entity_id_nospace"] = df["entity_id_norm"].apply(remove_spaces)
df["entity_id_tokens_sorted"] = df["entity_id_norm"].apply(sort_tokens_alpha)
df["entity_id_initials"] = df["entity_id_norm"].apply(initials_abbrev)

t_feat1 = time.perf_counter()
timings["string_feature_engineering"] = t_feat1 - t_feat0
logger.info("String feature engineering took %.3f seconds", timings["string_feature_engineering"])

# ALIASES / canonical name still commented out on purpose


# -------------------------------------------------------------------
# 3. Description embeddings with FastEmbed + multilingual-e5-small ONNX
# -------------------------------------------------------------------
t_emb0 = time.perf_counter()

TextEmbedding.add_custom_model(
    model="intfloat/multilingual-e5-small",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="intfloat/multilingual-e5-small"),
    dim=384,
    model_file="onnx/model.onnx",  # point this to your quantized ONNX if needed
)

embedding_model = TextEmbedding(model_name="intfloat/multilingual-e5-small")

desc_texts: List[str] = [
    "passage: " + (d or "") for d in df["description"].fillna("").astype(str).tolist()
]

# FastEmbed returns numpy arrays; convert to Python lists so DuckDB sees LIST
desc_embs = [e.tolist() for e in embedding_model.embed(desc_texts)]
df["description_embedding"] = desc_embs

t_emb1 = time.perf_counter()
timings["embedding_generation"] = t_emb1 - t_emb0
logger.info("Embedding generation took %.3f seconds", timings["embedding_generation"])


# -------------------------------------------------------------------
# 4. Splink settings – block on entity_type, lenient thresholds
# -------------------------------------------------------------------
db_api = DuckDBAPI()

# Embedding comparison via DuckDB's list_cosine_similarity on LIST columns
embedding_comparison = cl.CustomComparison(
    output_column_name="description_embedding",
    comparison_description="Cosine similarity on description embeddings",
    comparison_levels=[
        cll.NullLevel("description_embedding"),
        cll.CustomLevel(
            "list_cosine_similarity(description_embedding_l, description_embedding_r) >= 0.8",
            label_for_charts="High similarity (>= 0.8)",
        ),
        cll.CustomLevel(
            "list_cosine_similarity(description_embedding_l, description_embedding_r) >= 0.7",
            label_for_charts="Medium similarity (>= 0.7)",
        ),
        cll.ElseLevel(),
    ],
)

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        # 1) Normalised name
        cl.JaroWinklerAtThresholds(
            "entity_id_norm",
            score_threshold_or_thresholds=[0.97, 0.9, 0.8],
        ),
        # 2) No-space name
        cl.JaroWinklerAtThresholds(
            "entity_id_nospace",
            score_threshold_or_thresholds=[0.97, 0.9, 0.8],
        ),
        # 3) Sorted tokens
        cl.JaroWinklerAtThresholds(
            "entity_id_tokens_sorted",
            score_threshold_or_thresholds=[0.95, 0.85],
        ),
        # 4) Initials with thresholds (lenient)
        cl.JaroWinklerAtThresholds(
            "entity_id_initials",
            score_threshold_or_thresholds=[0.95, 0.85],
        ),
        # 5) Embedding-based similarity
        embedding_comparison,
    ],
    blocking_rules_to_generate_predictions=[
        # BLOCKING: only compare within same entity_type (your requirement)
        block_on("entity_type"),
    ],
    retain_intermediate_calculation_columns=True,
    em_convergence=0.01,
    # probability_two_random_records_match will be estimated from data
)


# -------------------------------------------------------------------
# 5. Initialise linker and TRAIN with your deterministic logic
# -------------------------------------------------------------------
t_linker0 = time.perf_counter()
linker = Linker(df, settings, db_api=db_api)
t_linker1 = time.perf_counter()
timings["linker_init"] = t_linker1 - t_linker0
logger.info("Linker initialisation took %.3f seconds", timings["linker_init"])

# Deterministic rule:
# same entity_type AND
# (strong name in ANY of the entity_id-derived fields OR strong embedding similarity)
deterministic_rules = ["""
    l.entity_type = r.entity_type
    AND (
        jaro_winkler_similarity(l.entity_id_norm, r.entity_id_norm) > 0.97
        OR jaro_winkler_similarity(l.entity_id_nospace, r.entity_id_nospace) > 0.97
        OR jaro_winkler_similarity(l.entity_id_tokens_sorted, r.entity_id_tokens_sorted) > 0.97
        OR jaro_winkler_similarity(l.entity_id_initials, r.entity_id_initials) > 0.97
        OR list_cosine_similarity(l.description_embedding, r.description_embedding) >= 0.8
    )
"""]

# Prior P(two random records match) from those high-precision rules
t_pmatch0 = time.perf_counter()
linker.training.estimate_probability_two_random_records_match(
    deterministic_rules,
    recall=0.5,  # your guess of how much of true matches these rules capture
)
t_pmatch1 = time.perf_counter()
timings["estimate_probability_two_random_records_match"] = t_pmatch1 - t_pmatch0
logger.info(
    "estimate_probability_two_random_records_match took %.3f seconds",
    timings["estimate_probability_two_random_records_match"],
)

# Estimate u (non-match) probabilities via random sampling
t_u0 = time.perf_counter()
linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
t_u1 = time.perf_counter()
timings["estimate_u_using_random_sampling"] = t_u1 - t_u0
logger.info(
    "estimate_u_using_random_sampling took %.3f seconds",
    timings["estimate_u_using_random_sampling"],
)

# EM training using entity_type-only blocking
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
output_lines: List[str] = []  # we’ll accumulate markdown here

# Optional: debug cosine similarity for the ADIB case
try:
    adib_emb = df[df["entity_id"] == "ADIB Group"]["description_embedding"].iloc[0]
    bank_emb = df[df["entity_id"] == "Abu Dhabi Islamic Bank"]["description_embedding"].iloc[0]

    con = duckdb.connect()
    sim = con.execute(
        "SELECT list_cosine_similarity(?, ?)", [adib_emb, bank_emb]
    ).fetchone()[0]
    debug_line = (
        f"DEBUG: Cosine similarity between **'ADIB Group'** and "
        f"**'Abu Dhabi Islamic Bank'**: `{sim:.4f}`"
    )
    print(debug_line)
    logger.info(debug_line)
    output_lines.append(debug_line)
except IndexError:
    msg = "DEBUG: ADIB Group or Abu Dhabi Islamic Bank not found in df; skipping sim debug."
    print(msg)
    logger.warning(msg)
    output_lines.append(msg)

t_inf0 = time.perf_counter()
pairwise_predictions = linker.inference.predict(
    threshold_match_weight=-5  # low to keep most candidate pairs
)
t_inf1 = time.perf_counter()
timings["inference_predict"] = t_inf1 - t_inf0
logger.info("inference.predict took %.3f seconds", timings["inference_predict"])

t_df0 = time.perf_counter()
preds = pairwise_predictions.as_pandas_dataframe()
t_df1 = time.perf_counter()
timings["pairwise_as_dataframe"] = t_df1 - t_df0
logger.info(
    "pairwise_predictions.as_pandas_dataframe took %.3f seconds",
    timings["pairwise_as_dataframe"],
)

# Columns to show for debugging
cols = [
    "match_weight",
    "match_probability",
    "bf_description_embedding",  # bayes factor for embedding comparison
]
for col in ["entity_type_l", "entity_id_l", "entity_type_r", "entity_id_r"]:
    if col in preds.columns:
        cols.insert(0, col)

top10 = preds.sort_values("match_weight", ascending=False).head(10)[cols]

print("\n--- Top 10 Pairwise Predictions (markdown) ---")
top10_md = top10.to_markdown(index=False)
print(top10_md)

output_lines.append("\n## Top 10 pairwise predictions\n")
output_lines.append(top10_md)

# Clustering
t_cluster0 = time.perf_counter()
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions,
    threshold_match_probability=0.5,  # tune this as you inspect results
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
timings["clusters_as_dataframe"] = t_clusters_df1 - t_clusters_df0
logger.info(
    "clusters.as_pandas_dataframe took %.3f seconds",
    timings["clusters_as_dataframe"],
)


# -------------------------------------------------------------------
# 7. Build merge plan (use raw entity_id for merged name)
# -------------------------------------------------------------------
merge_plans = []
for cluster_id, g in df_clusters.groupby("cluster_id"):
    if len(g) == 1:
        continue

    # Choose the longest raw entity_id as merged display name
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

print("\nSuggested merges (markdown):")
merge_md = merge_df.to_markdown(index=False)
print(merge_md)

output_lines.append("\n## Suggested merges\n")
output_lines.append(merge_md)


# -------------------------------------------------------------------
# 8. Add timing summary and save all results
# -------------------------------------------------------------------
SCRIPT_END = time.perf_counter()
timings["total_runtime"] = SCRIPT_END - SCRIPT_START
logger.info("Total script runtime: %.3f seconds", timings["total_runtime"])

# Add timing summary to markdown output
output_lines.append("\n## Timing summary (seconds)\n")
for name, sec in timings.items():
    output_lines.append(f"- **{name}**: {sec:.3f}")

# Write timings to separate file
with open(TIMING_PATH, "w", encoding="utf-8") as tf:
    tf.write("Timing summary (seconds)\n")
    for name, sec in timings.items():
        tf.write(f"{name}: {sec:.6f}\n")

logger.info("Timing summary written to %s", TIMING_PATH)

# Write markdown debug output
with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
    f.write("# Splink entity resolution debug output\n\n")
    f.write("\n".join(output_lines))

print(f"\nAll markdown output written to {OUTPUT_MD_PATH}")
logger.info("All markdown output written to %s", OUTPUT_MD_PATH)
logger.info("Log file written to %s", LOG_PATH)
