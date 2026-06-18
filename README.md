# AIML4OS_WP11_reconstruction

Given a set of enterprises, their properties, a link prediction function, 
derive a network of enterprises.

**WORK IN PROGRESS / IN HIGH FLUX**

```mermaid
flowchart TD

    A["📊 Input:<br>data-raw/data.parquet<br>data-raw/pairs.parquet"] --> S01

    subgraph GENERATION["Step 01: Candidate Generation"]
        S01["Generate Candidates<br>(from pairs or probabilistic)"]
    end

    S01 --> CAND["candidates.parquet"]

    subgraph SCORING["Step 02: Score Candidates"]
        S02["Load LightGBM Model<br>Predict Raw Probabilities"]
    end

    CAND --> S02
    S02 --> RAWPROB["raw_probabilities.parquet"]

    subgraph DEGREE["Step 03: Expected Degree"]
        S03["Estimate Expected Suppliers<br>(heuristic from TO/WAGES)"]
    end

    A --> S03
    S03 --> EXPECTED["expected_suppliers.parquet"]

    subgraph CALIB["Step 04: Calibration"]
        S04["Scale Raw Probabilities<br>to Match Expected Degree"]
    end

    RAWPROB --> S04
    EXPECTED --> S04
    S04 --> CALIBRATED["calibrated_probabilities.parquet"]

    subgraph RECONSTRUCT["Step 05: Reconstruct"]
        S05A["Step 05a: Sample Edges<br>(probabilistic or top-k)"]
        S05B["Step 05b: Prune Links<br>(degree + NACE exclusions)"]
        S05A --> S05B
    end

    CALIBRATED --> S05A
    EXPECTED --> S05A
    A --> S05B
    S05B --> RECON["reconstructed_network.parquet"]

    subgraph EVAL["Step 06: Evaluate"]
        S06["Compute Metrics<br>(edges, degree fit, MAE/RMSE)"]
    end

    RECON --> S06
    EXPECTED --> S06
    S06 --> SUMMARY["evaluation_summary.json<br>evaluation_user_degree_fit.parquet"]

    SUMMARY --> M["✅ Output Network"]

    style A fill:#e1f5ff
    style M fill:#c8e6c9
    style GENERATION fill:#fff3e0
    style SCORING fill:#fff3e0
    style DEGREE fill:#fff3e0
    style CALIB fill:#fff3e0
    style RECONSTRUCT fill:#fff3e0
    style EVAL fill:#fff3e0
```

## Run the Pipeline

Input data is read from `data-raw/`:

- `data-raw/data.parquet` (enterprise properties)
- `data-raw/pairs.parquet` (candidate pairs; preferred when available)

Run end-to-end:

```bash
python main.py
```

### Reconstruction Modes

Two reconstruction modes are available:

- `probabilistic` (default): weighted sampling without replacement.
- `topk`: deterministic top-k selection per user for reproducibility baselines.

Examples:

```bash
# Default probabilistic mode
python main.py --selection-mode probabilistic

# Deterministic baseline
python main.py --selection-mode topk
```

### Optional NACE Exclusions

Cross-sector links are allowed by default. You can optionally exclude specific user-NACE -> supplier-NACE combinations.

Template file:

- `data-raw/nace_exclusions.csv`

Expected columns:

- `user_nace`
- `supplier_nace`

Run with exclusions:

```bash
python main.py \
	--selection-mode probabilistic \
	--nace-exclusions data-raw/nace_exclusions.csv
```

### Outputs

Pipeline outputs are written to `data/`:

- `candidates.parquet`
- `raw_probabilities.parquet`
- `expected_suppliers.parquet`
- `calibrated_probabilities.parquet`
- `sampled_edges.parquet`
- `reconstructed_network.parquet`
- `evaluation_summary.json`
- `evaluation_user_degree_fit.parquet`

## Workflow Management with Snakemake

A `Snakefile` is provided for reproducible workflow orchestration. All pipeline steps are tracked as Snakemake rules with automatic dependency resolution.

### Installation

Install Snakemake (optional, if not already in your environment):

```bash
pip install snakemake
```

### Running with Snakemake

Default run (probabilistic mode):

```bash
snakemake --cores 1
```

Run with deterministic top-k baseline:

```bash
snakemake --cores 1 --config selection_mode=topk
```

Run with NACE exclusions:

```bash
snakemake --cores 1 --config nace_exclusions=data-raw/nace_exclusions.csv
```

Dry-run to visualize DAG:

```bash
snakemake -n --cores 1
```

### Configuration

Edit `config.yaml` to set defaults, or override individual settings at the command line:

```bash
snakemake --cores 1 \
  --config selection_mode=topk random_state=123 max_suppliers_per_user=30
```

Available config keys:

- `python`: Path to Python interpreter (default: `.venv/bin/python`)
- `data_raw_dir`: Input directory (default: `data-raw`)
- `data_dir`: Output directory (default: `data`)
- `model`: Path to LightGBM model (default: `models/model_LightGBM.pkl`)
- `selection_mode`: Reconstruction mode (default: `probabilistic`; options: `probabilistic`, `topk`)
- `nace_exclusions`: Path to NACE exclusion file (default: `""` disabled)
- `random_state`: Random seed (default: `42`)
- `max_suppliers_per_user`: Max outgoing degree (default: `25`)
- `max_users_per_supplier`: Max incoming degree (default: `500`)
