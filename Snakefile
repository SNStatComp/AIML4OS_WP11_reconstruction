from pathlib import Path

# Configurable parameters (override with: snakemake --config key=value)
PYTHON = config.get("python", str(Path(".venv/bin/python")))
DATA_RAW_DIR = config.get("data_raw_dir", "data-raw")
DATA_DIR = config.get("data_dir", "data")
MODEL_PATH = config.get("model", "models/model_LightGBM.pkl")
SELECTION_MODE = config.get("selection_mode", "probabilistic")
NACE_EXCLUSIONS = config.get("nace_exclusions", "")
RANDOM_STATE = int(config.get("random_state", 42))
MAX_SUPPLIERS_PER_USER = int(config.get("max_suppliers_per_user", 25))
MAX_USERS_PER_SUPPLIER = int(config.get("max_users_per_supplier", 500))

CANDIDATES = f"{DATA_DIR}/candidates.parquet"
RAW_PROBS = f"{DATA_DIR}/raw_probabilities.parquet"
EXPECTED = f"{DATA_DIR}/expected_suppliers.parquet"
CALIBRATED = f"{DATA_DIR}/calibrated_probabilities.parquet"
SAMPLED = f"{DATA_DIR}/sampled_edges.parquet"
RECONSTRUCTED = f"{DATA_DIR}/reconstructed_network.parquet"
EVAL_JSON = f"{DATA_DIR}/evaluation_summary.json"
EVAL_USER_FIT = f"{DATA_DIR}/evaluation_user_degree_fit.parquet"


def optional_nace_arg():
    return f"--nace-exclusions {NACE_EXCLUSIONS}" if str(NACE_EXCLUSIONS).strip() else ""


rule all:
    input:
        EVAL_JSON,
        EVAL_USER_FIT,


rule generate_candidates:
    input:
        enterprises=f"{DATA_RAW_DIR}/data.parquet",
    output:
        CANDIDATES,
    shell:
        (
            "{PYTHON} -m network_builder._01_generate_candidates "
            "--data-raw-dir {DATA_RAW_DIR} "
            "--output {output} "
            "--random-state {RANDOM_STATE}"
        )


rule predict_raw_probabilities:
    input:
        candidates=CANDIDATES,
        model=MODEL_PATH,
    output:
        RAW_PROBS,
    shell:
        (
            "{PYTHON} -m network_builder._02_predict_raw_probs "
            "--candidates {input.candidates} "
            "--output {output} "
            "--model {input.model}"
        )


rule estimate_expected_suppliers:
    input:
        enterprises=f"{DATA_RAW_DIR}/data.parquet",
    output:
        EXPECTED,
    shell:
        (
            "{PYTHON} -m network_builder._03_expected_suppliers_per_user "
            "--enterprises {input.enterprises} "
            "--output {output}"
        )


rule calibrate_probabilities:
    input:
        raw=RAW_PROBS,
        expected=EXPECTED,
    output:
        CALIBRATED,
    shell:
        (
            "{PYTHON} -m network_builder._04_calibrate_probs "
            "--raw-probabilities {input.raw} "
            "--expected {input.expected} "
            "--output {output}"
        )


rule reconstruct:
    input:
        calibrated=CALIBRATED,
        expected=EXPECTED,
        enterprises=f"{DATA_RAW_DIR}/data.parquet",
    output:
        sampled=SAMPLED,
        reconstructed=RECONSTRUCTED,
    params:
        nace_exclusions=lambda wc: optional_nace_arg(),
    shell:
        (
            "{PYTHON} -m network_builder._05_reconstruct "
            "--calibrated {input.calibrated} "
            "--expected {input.expected} "
            "--sampled {output.sampled} "
            "--output {output.reconstructed} "
            "--enterprises {input.enterprises} "
            "--selection-mode {SELECTION_MODE} "
            "--random-state {RANDOM_STATE} "
            "--max-suppliers-per-user {MAX_SUPPLIERS_PER_USER} "
            "--max-users-per-supplier {MAX_USERS_PER_SUPPLIER} "
            "{params.nace_exclusions}"
        )


rule evaluate:
    input:
        reconstructed=RECONSTRUCTED,
        expected=EXPECTED,
    output:
        summary=EVAL_JSON,
        user_fit=EVAL_USER_FIT,
    shell:
        (
            "{PYTHON} -m network_builder._06_evaluate "
            "--reconstructed {input.reconstructed} "
            "--expected {input.expected} "
            "--output-json {output.summary} "
            "--output-user-fit {output.user_fit}"
        )
