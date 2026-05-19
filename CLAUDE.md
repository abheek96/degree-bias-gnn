# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Research codebase investigating **why GCNs misclassify specific nodes**, focusing on structural graph factors (degree, neighbourhood purity, training-node proximity) and Jacobian-based influence signals. The primary claim is that GCN misclassification is systematic and predictable from node-level structural features.

## Running experiments

```bash
uv run main.py                          # uses config.yaml
uv run main.py --config my_config.yaml
uv run main.py --device cpu
```

The project uses `uv` (not `python` directly). All scripts are run with `uv run <script>.py`.

## Analysis scripts

All analysis scripts live in `analysis/` and are run from the repo root. See `STRUCTURE.md` for full documentation.

```bash
# Build per-test-node feature table + logistic regression
uv run analysis/node_feature_table.py --run 1 --no-influence --save-dir ./output
uv run analysis/node_feature_table.py --run 1 --save-dir ./output --plot-roc --shap

# Per-hop influence breakdown for specific nodes
uv run analysis/hop_influence.py --node-idx 1362 --run 1

# 1-hop influence aggregated by degree group (across all test nodes)
uv run analysis/1hop_influence_by_degree.py --run 1 --save-dir ./output

# Query / analyse / reachability by degree group
uv run analysis/degree_group.py --degree 5 --mode query
uv run analysis/degree_group.py --degree 5 --mode influence --run 1
uv run analysis/degree_group.py --all-degrees --mode reachability --save-dir ./output
```

Key flags for `analysis/node_feature_table.py`:
- `--run N` / `--checkpoint PATH` — model source (mutually exclusive)
- `--no-influence` — skip expensive Jacobian computation (~minutes per run)
- `--no-embeddings` — skip penultimate-layer embedding features
- `--features f1,f2,...` — restrict LR to specific features (e.g. `purity_1hop,purity_2hop`)
- `--plot-roc` — ROC/PR curves for degree-only / purity-only / full baselines
- `--shap` — OOF SHAP values + beeswarm and bar chart plots
- `--univariate-auroc` — feature-by-feature AUROC without LR

## Configuration system

`config.yaml` is the base config. Per-(model, dataset) overrides live in `configs/GCN_Cora.yaml`, `configs/GCN_PubMed.yaml`, etc. and are deep-merged on top at runtime. Both `main.py` and all analysis scripts apply this merge.

`split: random` uses `num_train_per_class: 20` with reproducible masks; `split: public` uses the dataset's fixed masks. Splits are cached in `dataset_cache/` as `.pt` files keyed by `{name}_{split}_{CC|noCC}_seed{seed}.pt` — delete cache if you change split parameters.

## Architecture

### Data flow

```
load_dataset (dataset.py)
  → CC filter (LargestConnectedComponents)
  → apply_split / load_or_create_split (dataset_utils.py)
  → PyG Data object with train/val/test masks
      → main.py: train loop → checkpoints saved under results/{exec}/checkpoints/
      → analysis scripts: load checkpoint → feature table → LR / SHAP
```

### Model structure

`models/gcn.py` — GCN layers are `nn.ModuleList` of `GCNConv` + final `nn.Linear` head. `get_intermediate(layer=k)` returns post-ReLU node embeddings after `k` message-passing layers (used for embedding similarity features). The `k_hops` used in influence analysis equals `num_layers - 1` (excludes the linear head).

`models/__init__.py` contains `MODEL_REGISTRY` and `get_model` factory — add new architectures here.

### Influence computation (`influence.py`)

`influence_distribution(model, data, node_x, k_hops)` computes the Jacobian-L1 influence of every node on `node_x` by evaluating `∂h_x^(k) / ∂h_y^(0)` on the k-hop induced subgraph. This is **expensive** (one Jacobian call per test node). `k_hop_subsets_exact` partitions the receptive field into exact hop rings (not cumulative).

### Feature table pipeline (`analysis/node_feature_table.py`)

Builds a CSV with one row per test node. Feature categories:
- **Structural**: degree, purity (1/2-hop), SPL to training nodes, training-node counts and ratios per hop
- **Influence**: Jacobian-L1 totals and fractions from same-class / diff-class nodes at hop 1 and 2
- **Embedding**: cosine similarity in penultimate GCN space between focal node and same-class / diff-class 1-hop neighbours (`emb_purity_delta`)

The LR uses 5-fold stratified CV with `class_weight="balanced"`. Target is `correct` (1=correct, 0=misclassified); PR-AUC is the primary metric (minority-class sensitive). SHAP values are OOF (each sample's SHAP from the fold that held it out), negated so positive = increases P(misclassified).

### Plotting (`plot_utils.py`)

Shared utilities: `_fig_w`, `_degree_axis`, `_save`, `_subdir`, `_BP_KWARGS`, `_ACC_COLOR`, `_PURITY_COLOR`. All analysis plots reuse these. Output goes to subdirectories under `--save-dir` (e.g. `output/shap/`, `output/roc_pr_curves/`).

## Git

Never include a Claude model as a co-author or mention Claude in git commit messages.

## Key documentation files

All research docs are in `docs/`:
- `docs/RESULTS.md` — main experimental results (PR-AUC is primary metric)
- `docs/RESULTS_EMBEDDINGS.md` — embedding feature results (preliminary; EPV concern flagged)
- `docs/TODO.md` — pending methodology fixes and planned analyses
- `docs/NOTES.md` — conceptual Q&A on metrics and plots
- `docs/COLUMNS.md` — full feature column definitions

`STRUCTURE.md` (root) — full map of every script, its purpose, why it was created, and its outputs.
