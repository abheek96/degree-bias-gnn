# Repository Structure

## Directory Layout

```
DegBias/
├── README.md
├── CLAUDE.md
├── STRUCTURE.md                  ← this file
├── config.yaml                   ← base experiment config
├── requirements.txt
│
├── # ── Library modules (imported by entry points and analysis scripts) ──
├── dataset.py
├── dataset_utils.py
├── train.py
├── test.py
├── utils.py
├── plot_utils.py
├── influence.py
├── logger.py
├── checkpoint_utils.py
│
├── # ── Primary entry points ──────────────────────────────────────────
├── main.py
├── generate_report.py
│
├── configs/                      ← per-(model, dataset) YAML overrides
│   ├── GCN_Cora.yaml
│   ├── GCN_CiteSeer.yaml
│   ├── GCN_PubMed.yaml
│   └── ...
│
├── models/
│   ├── __init__.py               ← MODEL_REGISTRY and get_model() factory
│   ├── gcn.py
│   ├── gat.py
│   ├── graphsage.py
│   ├── gcnii.py
│   └── mpnn.py
│
├── analysis/                     ← post-hoc analysis CLI scripts
│   ├── hop_influence.py
│   ├── 1hop_influence_by_degree.py
│   ├── degree_group.py
│   ├── node_feature_table.py
│   └── node_csv_export.py
│
├── tests/
│   └── test_graph_symmetry.py
│
└── docs/
    ├── COLUMNS.md
    ├── NOTES.md
    ├── RESEARCH.md
    ├── RESULTS.md
    ├── RESULTS_EMBEDDINGS.md
    └── TODO.md
```

---

## Library Modules (root-level)

| File | Purpose | Why created | Outputs |
|---|---|---|---|
| `dataset.py` | Loads PyG datasets (Cora, CiteSeer, PubMed, Amazon, etc.); applies LargestConnectedComponents filtering | Centralised dataset loading needed by both training and all analysis scripts | PyG `Data` object |
| `dataset_utils.py` | Caches and applies train/val/test splits; supports random and public splits, with optional `fix_test_to_public` flag | Reproducible splits shared across multiple scripts and runs | `Data` object with masks; `.pt` cache files in `dataset_cache/` |
| `train.py` | Single-epoch gradient step | Isolated from eval logic so it can be reused across main.py, generate_report.py, and analysis scripts | Scalar loss |
| `test.py` | `evaluate(model, data)` — accuracy on train/val/test splits | Isolated so every script uses the same evaluation code | Dict `{train, val, test}` |
| `utils.py` | Graph-structural feature computation: degree, k-hop purity, shortest-path distances, neighbourhood cardinality, labelling ratio, feature cosine similarity | Structural metrics are common across all plots and the feature table | Per-node NumPy arrays / PyTorch tensors |
| `plot_utils.py` | All plotting functions: accuracy vs degree, purity boxplots, influence disparity, SPL combined, neighbourhood cardinality, SHAP support | Centralised so `main.py` and `generate_report.py` produce identical figures | PNG files saved to `--save-dir` subdirectories |
| `influence.py` | Exact Jacobian-L1 influence computation: `influence_distribution()`, `k_hop_subsets_exact()`, `compute_influence_disparity_all()` | Expensive Jacobian computation isolated so it can be skipped or reused | Per-node float tensors of influence scores |
| `logger.py` | `setup_logger(log_dir, run_name)` — configures file + console logging | Consistent log format across all runs; log files archived with checkpoints | Logger instance; log files under `results/` |
| `checkpoint_utils.py` | Shared helpers for model building, training, and checkpoint loading: `set_seed`, `_deep_merge`, `_build_model`, `train_model`, `load_from_checkpoint`, `_resolve_run_checkpoint`, `load_cfg`, `load_data` | Extracted from the old `analyse_hop_influence.py` to remove the dual library/CLI role; all analysis scripts import from here | `(pred, model)` tuple; resolved checkpoint paths |

---

## Primary Entry Points

### `main.py`
Trains the configured GCN for `num_runs` independent seeds. For each run: saves a checkpoint, logs per-epoch metrics, and produces degree-bias plots (accuracy vs degree, purity, influence disparity, SPL, neighbourhood cardinality). Uses `logger.py` for structured file logging.

**Run:** `uv run main.py [--config my.yaml] [--device cpu]`
**Outputs:** `results/{exec_name}/checkpoints/run{N:02d}_seed{S}.pt`, log files, PNGs.

### `generate_report.py`
End-to-end pipeline: trains the model for `num_runs` runs, computes all five structural metrics, generates each plot to a temp directory, then assembles a letter-sized multi-page PDF with the plots and characterisation text. Each page has a structural, task-setting, and model-influence column.

**Run:** `uv run generate_report.py [--cfg my.yaml] [--out report.pdf] [--skip-influence]`
**Outputs:** `degree_bias_report.pdf` (or custom path).

---

## Analysis Scripts (`analysis/`)

All scripts are run from the repo root with `uv run analysis/<script>.py ...`. The `sys.path` insert at the top of each script ensures root-level modules are found regardless of working directory.

### `analysis/node_feature_table.py`
Builds a per-test-node feature CSV (structural + influence + embedding features), trains a 5-fold stratified logistic regression to predict misclassification, and optionally produces ROC/PR curves and SHAP plots.

**Created to:** Quantify which structural factors best predict whether a specific node will be misclassified (the central empirical claim of the project).

**Key flags:** `--run N` / `--checkpoint PATH`, `--multi-run`, `--subset-across-runs`, `--no-influence`, `--no-embeddings`, `--features f1,f2,...`, `--feature-selection`, `--plot-roc`, `--shap`, `--shap-nodes 1362,42`

**Outputs:** `{save_dir}/node_feature_table/{dataset}_{model}_node_features_seed{S}.csv`, optional SHAP PNGs, ROC/PR PNGs. `--save-dir` is inferred from the checkpoint path if not set.

**`--subset-across-runs`:** loads all `num_runs` checkpoints, reconstructs a full single-run feature table per run (shared topology features computed once via `_prepare_topology`, run-dependent influence/embedding features per checkpoint), runs the standard PR-AUC subset ablation (`_eval_subsets`) on each, and aggregates each subset's PR-AUC as mean ± std across runs (`_aggregate_subset_across_runs`). Distinct from `--multi-run`, which averages features and predicts misclassification *frequency* via Ridge/Spearman. Outputs `subset_comparison_across_runs.csv` (per-subset `pr_auc_mean`, `pr_auc_std`, per-run columns) and `subset_comparison_across_runs.png` (bar chart with ±std error bars).

---

### `analysis/hop_influence.py`
Per-hop Jacobian-L1 influence breakdown for one or more specific test nodes (given by `--node-idx`). Prints a table showing influence from same-class / diff-class training nodes at each hop, plus GCN-normalised edge weights.

**Created to:** Debug why a specific node is misclassified — which training nodes are driving its logit and from how many hops away.

**Key flags:** `--node-idx 1362 [2210 ...]`, `--run N` / `--checkpoint PATH`

**Outputs:** Logged table to stdout; no files written.

---

### `analysis/1hop_influence_by_degree.py`
Aggregates 1-hop Jacobian influence across all test nodes and groups by test-node degree. Produces two plots: side-by-side same/diff-class influence boxplots, and purity-by-hop lines with accuracy overlay.

**Created to:** Check whether the asymmetry between same-class and diff-class influence at hop 1 tracks with degree and purity (bridging structural features and learned influence).

**Key flags:** `--run N` / `--checkpoint PATH`, `--save-dir DIR`

**Outputs:** `{save_dir}/1hop_influence_by_degree/{prefix}_1hop_influence_by_degree_seed{S}.png` and `..._1hop_influence_balance...png`.

---

### `analysis/degree_group.py`
Unified script for querying and analysing test nodes by degree group. Three modes selected with `--mode`:

- **`query`** — list post-CC test-node indices for a given degree (no model needed; prints to stdout). Replaces the old `query_degree_group.py`.
- **`influence`** — find misclassified nodes in the degree range that have a higher-degree same-class training node in their k-hop field; run Jacobian influence analysis and print a ranked table. Replaces `analyse_degree_group.py`.
- **`reachability`** — for every (mis)classified test node in the range, classify it into one of three buckets (no training reachable / training but no same-class / same-class reachable) and log misclassification rates. Use `--all-degrees` to iterate over all unique degrees and produce stacked-bar plots. Replaces `analyse_degree1_reachability.py`.

**Created to:** Consolidate three closely related degree-group scripts that shared 40+ lines of setup boilerplate.

**Key flags:** `--degree D` / `--degree-min D --degree-max D`, `--mode {query,influence,reachability}`, `--all-degrees` (reachability only), `--save-dir DIR`, `--run N`

**Outputs:** Stdout (query/influence); optional PNGs to `{save_dir}/reachability/` (reachability + `--all-degrees`).

---

### `analysis/node_csv_export.py`
Exports node-level features for all nodes (not just test nodes) to CSVs and an Excel attribute dictionary. Runs `num_runs` independent training runs and aggregates predictions by majority vote.

**Created to:** Produce a flat tabular dataset for offline pandas / notebook analysis and for sharing with collaborators who don't want to run PyG.

**Key flags:** `--num-runs N`, `--out-dir DIR`

**Outputs:** `{out_dir}/all_nodes.csv`, `test_nodes.csv`, `train_nodes.csv`, `attribute_dictionary.xlsx`.

---

## Tests (`tests/`)

### `tests/test_graph_symmetry.py`
Seven regression tests verifying: graph symmetry (all edges have a reverse), `_khop_neighbors` returns the correct incoming-neighbor set at k=1, `inspect_node_aggregation` neighbor sets match `_khop_neighbors`, training-node sets are consistent between the two code paths, `get_khop_cardinality` agrees with BFS counts, and influence-analysis nodes lie outside the 1-hop table but inside the k-hop receptive field.

**Created to:** Catch regressions from the incoming-edge direction fix applied to `_khop_neighbors` early in the project.

**Run:** `uv run tests/test_graph_symmetry.py`
