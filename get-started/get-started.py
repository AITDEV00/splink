import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

# 1. Backend
db_api = DuckDBAPI()

# 2. Demo data (has first_name, surname, dob, city, email, unique_id, etc.)
df = splink_datasets.fake_1000

# 3. Settings: link_type, comparisons, blocking
settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.JaroWinklerAtThresholds("first_name", [0.9, 0.7]),
        cl.JaroAtThresholds("surname", [0.9, 0.7]),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
            datetime_metrics=["year", "month"],
            datetime_thresholds=[1, 1],
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ],
    # df already has "unique_id" so we can leave this out,
    # but I'm making it explicit so it's obvious:
    unique_id_column_name="unique_id",
)

# 4. Build linker
linker = Linker(df, settings, db_api)

# 5. Training
# 5a. Prior: P(two random records match)
linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "surname")],
    recall=0.7,
)

# 5b. u-probabilities (non-match)
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# 5c. m-probabilities (match) via EM with a blocking rule
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob")
)

# 6. Inference (pairwise links)
pairwise_predictions = linker.inference.predict(threshold_match_weight=-10)

print("Top pairwise predictions")
print(
    pairwise_predictions.as_pandas_dataframe()
    .sort_values("match_weight", ascending=False)
    .head(10)
)

# 7. Clustering
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions,
    threshold_match_probability=0.95,
)

df_clusters = clusters.as_pandas_dataframe()
print("\nSample clustered output")
print(
    df_clusters[["cluster_id", "unique_id", "first_name", "surname", "dob", "city"]]
    .sort_values(["cluster_id", "unique_id"])
    .head(20)
)
