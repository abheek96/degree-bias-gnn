# DegBias — Degree Bias in Graph Neural Networks

Experimental framework for studying how node degree affects GNN classification accuracy.
Supports multi-run experiments across random seeds with structured logging, reproducible
dataset splits, and degree-stratified accuracy plots.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers `torch`, `torch-geometric`, `pyyaml`, `numpy`, `tqdm`, and
`matplotlib`.

---

## Project structure

```
DegBias/
├── main.py            # Entry point — orchestrates runs, logging, and results
├── train.py           # Single-epoch training step
├── test.py            # Evaluation (train / val / test accuracy)
├── dataset.py         # Dataset loading and CC filtering (load_dataset)
├── dataset_utils.py   # apply_split, get_accuracy_deg, get_str_info, make_deg_groups
├── plot_utils.py      # plot_acc_vs_degree and supporting helpers
├── logger.py          # setup_logger — file + console logging
├── config.yaml        # Default experiment configuration
├── models/
│   ├── __init__.py    # MODEL_REGISTRY and get_model factory
│   ├── gcn.py         # GCN
│   ├── gat.py         # GAT
│   └── graphsage.py   # GraphSAGE
└── results/           # Auto-created; one subdirectory per execution
```

---

## Configuration (`config.yaml`)

```yaml
seed: 42            # Base random seed; run i uses seed + i - 1
num_runs: 5         # Number of independent runs per execution
device: cuda        # cuda | cpu  (auto-detected if omitted on CLI)
results_dir: ./results
split: random       # random | public

dataset:
  name: Cora        # Cora | CiteSeer | PubMed | Computers | Photo | CS | Physics | WikiCS
  root: ./data
  use_cc: true      # Restrict to the largest connected component
  num_train_per_class: 20   # (random split only)
  num_val: 500              # (random split only)
  num_test: 1000            # (random split only)

model:
  name: GCN         # GCN | GAT | GraphSAGE
  hidden_dim: 512
  num_layers: 2
  dropout: 0.0

train:
  lr: 0.001
  weight_decay: 5e-4
  epochs: 200
  patience: 0       # Early-stopping patience; 0 = disabled

plot:
  acc_vs_degree: true
  save: true
  show: false
```

---

## Running an experiment

```bash
python main.py                          # uses config.yaml
python main.py --config my_config.yaml  # custom config
python main.py --device cpu             # override device
```

### What happens at runtime

1. An **execution directory** is created under `results_dir`:
   ```
   results/Cora_GCN_random_CC_03Mar2026_1430/
   ```
   The name encodes dataset, model, split type, CC flag, and timestamp.

2. **Dataset loading** — done once before the run loop.
   - If `use_cc: true`, the graph is restricted to its largest connected component.
   - If `split: random`, the train/val/test masks are created once with `seed`
     and are **shared across all runs** — only model initialisation changes per run.
   - If `split: public`, the masks provided by the dataset are used as-is.

3. **Per-run loop** — for run `i`, seed = `base_seed + i − 1`:
   - A dedicated log file and subdirectory are created:
     ```
     results/.../run01_seed42/run01_seed42.log
     results/.../run02_seed43/run02_seed43.log
     ```
   - The model is re-initialised and trained from scratch.
   - Best val/test accuracy is tracked with optional early stopping.

4. **Summary** — after all runs, aggregated mean ± std are written to:
   ```
   results/.../summary/summary.log
   ```

---

## Output directory layout

```
results/
└── Cora_GCN_random_CC_03Mar2026_1430/     ← execution directory
    ├── run01_seed42/
    │   └── run01_seed42.log
    ├── run02_seed43/
    │   └── run02_seed43.log
    ├── ...
    └── summary/
        └── summary.log
```

Plots (when enabled) are saved into the execution directory alongside the run folders.

---

## Degree-bias analysis and plotting

### Computing per-degree accuracy

`get_accuracy_deg` in `plot_utils.py` maps test-node predictions to node degrees:

```python
from torch_geometric.utils import degree
from plot_utils import get_accuracy_deg

# deg: degree of each test node (from original or CC graph)
deg  = degree(data.edge_index[1], data.num_nodes)[data.test_mask]
pred = model(data.x, data.edge_index).argmax(dim=1)[data.test_mask]
true = data.y[data.test_mask]

result = get_accuracy_deg(deg, pred, true)
# result = {degree_int: {'preds': tensor, 'labels': tensor, 'acc': float}, ...}
```

### Plotting

```python
from plot_utils import plot_acc_vs_degree

# run_results: list of get_accuracy_deg dicts, one per run
plot_acc_vs_degree(
    run_results,
    cfg,                    # config dict (for titles and labels)
    save_dir=exec_dir,      # directory to save PDFs
    show=False,             # set True to open interactive windows
)
```

**Single run** → one PDF:

| File | Content |
|---|---|
| `acc_vs_degree_single_run.pdf` | Box plot per degree — per-node correctness (0/1) distribution across all test nodes of that degree. Colour encodes degree magnitude (YlOrRd). |

**Multiple runs** → two PDFs:

| File | Content |
|---|---|
| `acc_vs_degree_across_runs.pdf` | Box plot per degree — distribution of *per-run mean accuracy* across seeds. Wide boxes signal seed-sensitive degrees. |
| `acc_vs_degree_per_run.pdf` | Grouped box plot — for each degree, one box per run (tab10 colours) showing the node-level accuracy distribution within that run. |

All figures carry the full experimental context in their title: dataset, model, split type,
and CC flag.

---

## Structural neighbourhood features

`make_deg_groups` and `get_str_info` in `dataset_utils.py` enrich the data object with
degree-based structural attributes:

```python
from dataset_utils import make_deg_groups

data = make_deg_groups(data, n_groups=4)
# Adds to data:
#   data.deg        — raw degree per node
#   data.deg_labels — degree normalised to [0, 1]
#   data.deg_group  — integer bucket in {0, …, n_groups−1} (equal-width bins)
#   data.group1     — total length-1 walk count per node (= degree)
#   data.group2     — total length-2 walk count per node (structural richness)
```

> **Note:** `make_deg_groups` internally converts the graph to a dense N×N adjacency
> matrix for the walk computation — this is O(N²) in memory and O(N³) in compute.
> Avoid on large graphs (N > ~5 000).

---

## Supported datasets

| Name | Source | Notes |
|---|---|---|
| Cora | Planetoid | Citation network |
| CiteSeer | Planetoid | Citation network |
| PubMed | Planetoid | Citation network |
| Computers | Amazon | Co-purchase graph |
| Photo | Amazon | Co-purchase graph |
| CS | Coauthor | Co-authorship graph |
| Physics | Coauthor | Co-authorship graph |
| WikiCS | WikiCS | Wikipedia hyperlink graph |

## Supported models

| Name | Architecture |
|---|---|
| GCN | Graph Convolutional Network (Kipf & Welling, 2017) |
| GAT | Graph Attention Network (Veličković et al., 2018) |
| GraphSAGE | Inductive Representation Learning (Hamilton et al., 2017) |
