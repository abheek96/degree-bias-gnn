# Degree Bias — Conceptual Q&A

A running log of questions and answers about the metrics, plots, and model behaviour in this project.

---

## Graph Structure & Metrics

### What is Shortest Path Length (SPL)?

SPL between two nodes is the minimum number of edges you need to traverse to get from one node to the other.
In this project, for each test node we compute the **average SPL to all training nodes** (and separately, to same-class training nodes only) using BFS from every training node outward.
A test node with high average SPL is structurally far from the labelled part of the graph — it receives less direct signal from training nodes during message passing.

### How is the SPL computed in the code?

`get_avg_spl_to_train` runs a BFS from each training node and records the shortest distance to every reachable node.
After all training-node BFS passes, each node accumulates its distances and divides by the number of training nodes that could reach it.
`get_avg_spl_to_same_class_train` does the same but only accumulates distances from training nodes that share the same class label as the target node.

### What is the labelling ratio?

For a test node, the labelling ratio is whether it has **at least one training node as a direct (1-hop) neighbour**.
It is a binary indicator per node (true/false), computed as: `(adjacency matrix × training mask) > 0`.
A test node with no labelled neighbours relies entirely on indirect signal propagated through non-training nodes.

### What does "labelling ratio for all nodes" mean?

The raw `get_labelling_ratio` function computes the indicator for every node in the graph.
In the analysis we then slice it to test nodes only (`[data.test_mask.cpu()]`) because we only care about whether the unlabelled nodes we evaluate on have labelled neighbours — not whether training nodes are adjacent to other training nodes.

---

## Plots

### What do the `acc_vs_degree_by_layers` plots show?

For each (model, layer-count) combination, each test node is assigned to a degree group.
Within each degree group, accuracy is averaged across runs.
The plot shows how that mean accuracy changes as the number of layers increases — one line per degree group.
This lets you see whether adding depth helps or hurts nodes of a specific degree.

### Is the y-axis on the trend plot (delta accuracy) the slope of the line?

Yes. The `plot_acc_trend_by_degree` plot fits a linear regression of accuracy vs. layer count for each degree group and plots the resulting slope.
A positive slope means accuracy improves with more layers for that degree group; a negative slope means it degrades.

### What do the `purity_vs_degree` plots show?

**Neighbourhood purity** for a node at k-hop expansion is the fraction of its k-hop neighbours that share the node's own class label.
A degree-1 node has exactly one neighbour, so its purity is either 0 or 1.
A high-degree node has many neighbours, so its purity converges toward the global class distribution.

The plot shows the mean purity per degree group for a fixed k.
Multiple k values are plotted to show how purity changes as the neighbourhood expands.

### What does the delta-purity plot show?

It plots `purity(k_max) − purity(k_min)` per degree group — i.e., how much the average purity changes as the neighbourhood expands from the smallest to the largest k.
A negative delta means that expanding the neighbourhood dilutes the class signal (more heterogeneous neighbours further out).
This tends to be more pronounced for high-degree nodes whose wider neighbourhoods span more class boundaries.

---

## Feature Similarity Delta

### What is the feature similarity delta?

For each test node `v` that has at least one same-class training node as a direct (1-hop) neighbour, two cosine similarity scores are computed:

```
sim_raw(v) = mean cosine_sim( x_v,   x_u )   for u ∈ same-class train ∩ N₁(v)
sim_h1(v)  = mean cosine_sim( h_v¹,  h_u¹ )  for the same u

delta(v)   = sim_h1(v) − sim_raw(v)
```

`x` is the raw input feature vector (1433-dim bag-of-words for Cora).
`h¹` is the representation after the first GCNConv layer + ReLU — obtained via `model.get_intermediate(layer=1)` — before the linear classification head.

A **positive delta** means message passing brought `v` closer to its same-class training neighbours in representation space: the neighbourhood signal reinforced class-consistent features.
A **negative delta** means message passing pulled `v` away: diff-class neighbours in N₁(v) introduced feature-space noise into `h¹`, analogous to what low label purity captures structurally.

### Why use cosine similarity and not Euclidean distance?

Cosine similarity measures the **angle** between two vectors, ignoring their magnitude. This is appropriate here for two reasons:

1. After ReLU and degree normalisation, the scale of `h¹` varies substantially across nodes — magnitude differences would dominate a Euclidean comparison and obscure the directional alignment that actually matters for classification.
2. The raw input features are binary bag-of-words — two papers on the same topic may have very different numbers of keywords (different magnitudes) but still point in similar directions. Cosine similarity captures that semantic alignment directly.

### How does feature similarity delta complement purity and influence disparity?

The three metrics attack the same underlying question from different angles:

| Metric | What it measures | Space |
|---|---|---|
| Purity | Fraction of same-class structural neighbours | Graph topology + labels |
| Feature similarity delta | Whether aggregation preserved same-class feature alignment | Input / hidden feature space |
| Influence disparity | Whether same-class training nodes dominate gradient flow | Model weights / Jacobian |

Purity tells you the structural precondition: how many diff-class nodes are present.
Feature similarity delta tells you whether those diff-class nodes actually corrupt the representation — a node could have low purity but still have a good `h¹` if its diff-class neighbours happen to have dissimilar features.
Influence disparity tells you whether the corruption propagates all the way to the model's decision.

A node where all three are negative (low purity, negative delta, negative influence disparity) is the strongest possible instance of degree bias: structural imbalance, representation corruption, and gradient-flow imbalance all converge on the same node.

### Why are nodes with no same-class training neighbour excluded?

The metric specifically measures whether message passing preserves or degrades similarity *to same-class training nodes that are directly reachable*. If there is no same-class training node in N₁(v), there is no reference signal to compare against — the metric is undefined for that node, not zero.
These nodes are separately captured by the labelling ratio and SPL metrics, which measure the absence of proximate same-class signal.

### What does `model.get_intermediate(layer=1)` return exactly?

It runs the input `x` through the first GCNConv layer only and applies ReLU, stopping before subsequent layers and the linear head. For the default 2-layer GCN (`num_layers=2`), `self.layers[:-1]` contains exactly one GCNConv, so `layer=1` is the only valid value. It returns a `(N, hidden_dim)` tensor detached from the compute graph, with `model.eval()` set to suppress dropout.

---

## Influence Analysis

### How does the influence analysis work?

It computes the **exact influence distribution** of Definition 3.1:

```
I(x, y) = Σ_{i,f}  |∂h_x^(k)[i] / ∂h_y^(0)[f]|
I_x(y)  = I(x, y) / Σ_z I(x, z)
```

`h_x^(k)` is the final-layer embedding of node x.
`h_y^(0)` is the input feature vector of node y.
The Jacobian is computed exactly via `torch.autograd.functional.jacobian`.
Summing the absolute values over all output dimensions `i` and all input feature dimensions `f` gives a scalar "how much does y's input affect x's output".
Normalising over all nodes gives a probability distribution over the graph — `I_x(y)` is the fraction of x's total sensitivity attributable to node y.

### How are `same_class_influence` and `diff_class_influence` reported?

Two levels of reporting are produced:

**Raw scores** — the sum of `I_x(y)` over same-class (or diff-class) training nodes in the receptive field. These are fractions of the full-graph influence (which sums to 1 over all N nodes), so they are typically very small (e.g. `1e-7` for same, `5e-2` for diff).

**Normalised scores** — the raw scores divided by `total_train_influence = same + diff`. These sum to 1 between the two groups and are the primary values used for comparison and plotting. A normalised value of `same=0.04, diff=0.96` directly reads as "96% of the influence from training nodes in the receptive field comes from diff-class nodes".

The log prints both:
```
raw:  same=1.2e-07  diff=5.1e-02  total_train=5.1e-02
norm: same=0.0000   diff=1.0000   (fraction of training-node influence)
```

Plots use the normalised scores so bars across different nodes are on a common [0, 1] scale.

### Per-hop breakdown (`analyse_hop_influence.py`)

The aggregate table above collapses all hops in the receptive field into a single same/diff comparison, normalised by **training-node total only**. For deeper models (k_hops ≥ 2) this hides *where* the signal comes from and ignores the share routed through non-training nodes.

`analyse_hop_influence.py` breaks the same Jacobian-L1 computation down per hop. For each hop `i` in `[0, k_hops]` it reports:

- `total_inf[i]` — sum of `I_x(y)` over **all** nodes exactly i hops from x (training or not). Derived from one Jacobian call via `k_hop_subsets_exact` + `influence_distribution`, so numerically identical to `jacobian_l1_agg_per_hop` but without re-running the forward pass.
- `same_inf[i]` / `diff_inf[i]` — raw (unnormalised) sums over same-class / diff-class training nodes at hop i.
- Fractions use the hop's **total** as the denominator (not the training total), so `same/tot + diff/tot + non_train_frac = 1` at each hop.
- Cardinalities: `#same_tr`, `#diff_tr`, `#non_tr` summing to `|S_i|` (the hop set).

`k_hops = num_layers - 1` (final linear layer does no message passing), so `hop 0` is self-influence of node x's own features and `hop k_hops` is the outer shell.

Model source: `--checkpoint PATH` loads a saved state_dict; `--run N` auto-locates the most recent `results/{dataset}_{model}_{split}_{CC|noCC}_*/checkpoints/run{N:02d}_*.pt`; neither retrains from scratch. `main.py` now writes these checkpoints after every run under `{exec_dir}/checkpoints/` (gated by `train.save_checkpoints`, default True).

### What is `influence_top_n` / why was it there?

It was a cap on the number of nodes analysed per degree value, to limit compute when many nodes share a degree.
It has since been removed — the current logic selects exactly one node per requested degree (the misclassified one, or the only candidate if there's just one).

### What do `same_train` and `diff_train` mean in the log? Are they scores?

No, they are **counts**:

- `same_train=4` — there are **4 training nodes** of the **same class** as the selected node inside its k-hop receptive field.
- `diff_train=11` — there are **11 training nodes** of a **different class** inside the same receptive field.

`same_class_influence` and `diff_class_influence` are the actual scores — the sum of `I_x(y)` over the respective groups, further normalised by the total training-node influence so they sum to 1 between them (see above).

### Why is `same_class_influence ≈ 0` when `same_train=4`?

The 4 same-class training nodes exist in the neighbourhood and were trained normally — the zero influence is not a training failure.
It means the **trained model's weights** have converged to a state where essentially no gradient flows from those nodes back to the target node. Concrete causes:

1. **Dead ReLU paths.** If any ReLU along the message-passing path from a same-class training node to the target outputs 0 for this particular input, the Jacobian through that path is exactly 0. With multiple layers, there are many such potential choke points.

2. **Aggregation dilution.** Node 1362 has degree 22. GCN normalises each neighbour's contribution by the product of degree normalisations. A same-class node that is 2 hops away gets diluted by two normalisation factors, making its contribution negligible relative to the 11 closer diff-class nodes.

3. **Learned weight suppression.** After training, the weight matrices may project the same-class neighbours' features into a direction that has near-zero magnitude for this node's final representation.

The fact that `same_class_influence ≈ 0` for a **misclassified** node with 4 same-class and 11 diff-class training neighbours is precisely the anomaly this analysis is designed to surface: for this specific node, the model cannot effectively route same-class information because diff-class nodes dominate aggregation numerically. Note that this is an anomaly — most high-degree nodes are correctly classified. Node 1362 represents a case where the neighbourhood class imbalance is severe enough, combined with the other compounding factors, to tip the model into misclassification despite the node's high connectivity.

### Case study: node 387 (degree=16) — 2-hop training nodes and low-degree diff-class domination

Node 387 illustrates a failure mode distinct from node 1362. It is misclassified despite having 11 of 17 one-hop neighbours sharing its class (class 4).

**1-hop aggregation neighbourhood:**

```
    neighbor  degree  in_train_set  same_class  correct_pred  edge_weight
1       1372     1.0         False        True          True     0.171499
2        640     2.0         False        True         False     0.140028
3       1903     2.0         False        True         False     0.140028
4        172     3.0         False        True          True     0.121268
5        475     3.0         False        True         False     0.121268
6         32     4.0         False        True         False     0.108465
7         76     4.0         False        True         False     0.108465
8       1486     5.0         False        True          True     0.099015
9       1749     9.0         False        True         False     0.076697
10      1861    16.0         False        True          True     0.058824
11       387    16.0         False        True         False     0.058824   ← self
12      1904     2.0         False       False         False     0.140028
13      1534     4.0         False       False         False     0.108465
14      1901     4.0         False       False          True     0.108465
15       724     5.0         False       False         False     0.099015
16       791     5.0         False       False         False     0.099015
17      1669     5.0         False       False          True     0.099015
```

Every 1-hop neighbour is a non-training node. The aggregation table therefore shows no direct supervision signal.

**Influence analysis (2-hop Jacobian):**

```
influence: node 387  degree=16  same_train=1  diff_train=2
raw:  same=4.1427e-03  diff=5.2353e-02  total_train=5.6496e-02
norm: same=0.0733  diff=0.9267  (fraction of training-node influence)
  same_train node 2221   deg=3   raw=4.1427e-03  norm=0.0733
  diff_train node 456    deg=3   raw=4.1427e-03  norm=0.0733
  diff_train node 2248   deg=2   raw=4.8210e-02  norm=0.8533
```

**Key observations:**

1. **Training nodes are at 2-hop, not 1-hop — because the run used `num_layers=3`.** `inspect_node_aggregation` always shows 1-hop neighbours (hardcoded: it finds all edges where the destination equals `node_idx`). The influence analysis uses `k_hops = num_layers - 1` as its receptive-field radius, so for `num_layers=3` it searches 2 hops out. With the default `num_layers=2` (`k_hops=1`) the two analyses cover the same single hop; if no training nodes appeared in the 1-hop table, the influence analysis would have skipped the node with a warning. The fact that 2221, 456, and 2248 were found — and that none are in the 1-hop table — confirms `num_layers=3` was active for that run. This is a concrete demonstration of why `num_layers` matters for both what the model can aggregate and which training nodes the influence analysis can surface.

2. **A single low-degree diff-class training node dominates.** Node 2248 (degree=2, different class) accounts for 85.3% of all training-node influence. Because it has degree 2, it gets high edge weights in both aggregation hops, and that amplification compounds across layers — the model ends up using it as the primary anchor for node 387's prediction.

3. **The same-class training node is swamped.** Node 2221 (degree=3, same class) has equal raw influence to diff-class node 456 (4.14e-3 each), but together they contribute only 14.7% against node 2248's 85.3%. The structural cause is node 2248's very low degree giving it disproportionately high normalised edge weights at each hop.

### Case study: node 1894 (degree=40) — the curse of high degree

Node 1894 illustrates a third distinct failure mode. Unlike nodes 1362 and 387, the training-node influence here is *correctly* same-class dominated — yet the node is still misclassified.

**Influence analysis:**
```
influence: node 1894  degree=40  same_train=2  diff_train=1
raw:  same=1.4010e-02  diff=1.3318e-03  total_train=1.5342e-02
norm: same=0.9132  diff=0.0868
  same_train node 1828   deg=3    hop=1   norm=0.5895
  same_train node 184    deg=10   hop=2   norm=0.3237
  diff_train node 271    deg=78   hop=2   norm=0.0868
```

**1-hop aggregation neighbourhood (40 neighbors):**
- Same-class: 10 (rows 1–10, including 1 training node — 1828)
- Diff-class:  30 (rows 12–41, all non-training)
- Neighbourhood purity: **25% same-class**

**Why it fails — the curse of high degree:**

1. **Diluted edge weights.** Node 1894 has degree=40. Every edge weight is divided by `sqrt(40 × deg_neighbor)`. Even node 1828 (the best-placed same-class training node: deg=3, hop=1) gets edge weight `≈ 0.078`. High degree penalises all incoming signals, including the most informative ones.

2. **Noise from 30 diff-class non-training neighbors.** The training-node influence score (91.3% same-class) only measures the relative influence *among training nodes*. It does not account for the 30 diff-class non-training neighbors that are simultaneously aggregated into node 1894's representation. Each of those 30 nodes sends a diff-class feature vector through the GCN — the aggregate `h^(1)` for node 1894 is the weighted sum of all 40 neighbors, 30 of whom push it toward the wrong class.

3. **Training influence ≠ total representational influence.** A node can have 91% of its *training-node* influence from same-class sources and still be misclassified if the non-training neighborhood is overwhelmingly diff-class. The training-node influence metric is a necessary but not sufficient condition for correct classification.

**The curse of high degree** (proposed term): high-degree nodes face a structural disadvantage that compounds across the aggregation:
- High degree → lower per-edge weight for every neighbor, including same-class training nodes
- High degree → more neighbors to aggregate, increasing the probability of a large diff-class non-training majority
- High degree → lower neighbourhood purity (hub nodes span more class boundaries)
- Together these dilute the same-class training signal and flood the representation with class-boundary noise, even when the training-node influence distribution looks correct

This is distinct from nodes 1362 and 387, where diff-class *training* nodes dominated. Here, same-class training influence is healthy — the failure comes from the non-training neighbourhood. High degree makes a node structurally susceptible to this form of noise even when its labeled anchors are well-placed.

**Contrast with node 1362.** Node 1362 (degree=22) had `same_train=4`, `diff_train=11` — many training nodes of both classes in-neighbourhood, but same-class influence ≈ 0 due to dead ReLU paths / learned weight suppression. Node 387 is simpler structurally: it has one same-class training anchor that is correctly active, but a single low-degree wrong-class node overwhelms it purely through degree-normalisation arithmetic. These are two distinct mechanisms by which diff-class influence can dominate.

### Case study: node 1305 (degree=2) — high-degree same-class training neighbour, yet dominated by a 2-hop diff-class node

Node 1305 illustrates a failure mode specific to **low-degree test nodes adjacent to a high-degree same-class training node**. It is the only degree-2 misclassified test node with this configuration (found via `analyse_degree_group.py --degree 2`).

**Qualifying condition:** node 1305 (deg=2, class=3) has node 1555 (deg=19, same class) as a direct 1-hop training neighbour.

**1-hop aggregation neighbourhood:**

```
   neighbor  degree  hop  in_train_set  same_class  correct_pred  edge_weight
1      1555      19    1          True        True          True     0.129099
2       536       5    1         False       False          True     0.235702
```

Only 2 neighbours. Node 1555 is the single same-class training anchor; node 536 is diff-class and non-training. Despite node 536 being correctly predicted itself, it pushes a diff-class representation into node 1305's aggregation.

**Influence analysis:**

```
influence: node 1305  degree=2  same_train=2  diff_train=1
raw:  same=7.8450e-02  diff=4.4734e-02  total_train=1.2318e-01
norm: same=0.6369  diff=0.3631  (fraction of training-node influence)
  same_train node 1555   deg=19  hop=1  ew=0.129099  norm=0.4887
  same_train node 456    deg=3   hop=2  ew=N/A        norm=0.1482
  diff_train node 2303   deg=6   hop=2  ew=N/A        norm=0.3631
```

**Key observations:**

1. **Same-class influence nominally dominates (63.7% vs 36.3%), yet the node is still misclassified.** This is a case where training-node influence alone does not determine the outcome — the diff-class non-training neighbour 536 (edge weight 0.236) directly corrupts the aggregated representation at hop=1, and that is not captured in the training-node influence score.

2. **The high-degree same-class training node (1555, deg=19) pays a steep edge-weight penalty.** Its direct edge weight is only 0.129 (`1/sqrt((2+1)×(19+1)) ≈ 0.129`). By contrast, diff-class neighbour 536 (deg=5) gets weight 0.236 (`1/sqrt((2+1)×(5+1)) ≈ 0.236`) — nearly twice as strong a signal, despite being diff-class and non-training. The low-degree test node is structurally penalised for being connected to a high-degree training anchor.

3. **A 2-hop diff-class training node (2303, deg=6) contributes 36.3% of training-node influence** — more than node 456 (same class, deg=3, hop=2) at 14.8%. This mirrors the node 387 failure mode: a wrong-class node reaches through the graph more effectively than the right-class anchor because of path structure or degree.

4. **The correct prediction of neighbour 536 does not protect node 1305.** Node 536 is correctly predicted as its own class, meaning the GCN has learned a good representation for it — but that representation is still diff-class, and when aggregated into node 1305, it shifts the embedding toward the wrong class regardless.

**Contrast with node 1894.** Node 1894 failed despite healthy same-class training influence (91%) because of an overwhelmingly diff-class *non-training* neighbourhood (30 of 40 neighbours). Node 1305 fails with only 2 neighbours — one same-class training node (strongly placed, but penalised by high degree) and one diff-class non-training node (lower degree, higher edge weight). The failure here is structural arithmetic: being degree-2 and connected to a high-degree same-class training node means that neighbour's signal is inherently weaker per-edge than a lower-degree diff-class neighbour would be.

### Does `same_class_influence ≈ 0` mean those training nodes weren't trained?

No. All training nodes participate in the loss and gradient updates as usual.
The influence score is a property of the **forward pass of the trained model**, not of the training procedure.
It measures how much the trained model *uses* information from each neighbour when making a prediction — not whether that neighbour was part of training.
A training node with zero influence simply means the model has learned (or failed to learn) weights such that this node's features do not percolate meaningfully to the target node's representation.

### How does this influence measure differ from RawlsGCN's?

RawlsGCN (Theorem 1) also uses a gradient-based influence measure, but it is a fundamentally different quantity operating at a different stage of computation.

**RawlsGCN's influence (Theorem 1)** computes the gradient of the *training loss* with respect to the weight parameters `W^(l)`:

```
∂f / ∂W^(l) = Σ_i  deg_Â(i) · V̂_i^(col)
```

The gradient of the loss w.r.t. the weights decomposes as a weighted summation of per-node influence matrices `V̂_i^(col)`, where each node i's contribution is scaled by its degree in the renormalized graph Laplacian `Â = D^{-1/2}(A+I)D^{-1/2}`:

- `deg_Â(i) = Σ_j Â_{ij}` — the row-wise sum of Â for node i
- `V̂_i^(col)` — the column-wise influence matrix of node i, derived from the hidden embeddings `H^(l-1)` and the upstream gradient `∂f/∂t_i`
- `Ĥ^(l) = Â · H^(l-1) · W^(l)` — the node embeddings before nonlinear activation

This means high-degree nodes contribute proportionally more to the gradient update of the weights. The model is effectively trained more on high-degree nodes and less on low-degree nodes — the weight update disproportionately serves the high-degree population.

**This repository** computes the Jacobian of a *specific test node's output* with respect to each other node's input features:

```
I(x, y) = Σ_{i,f} |∂h_x^(k)[i] / ∂h_y^(0)[f]|
```

This is a **local, pairwise** measure — it captures how much node y's features steer node x's final representation. It is computed numerically through the full trained model.

The key distinctions:

| | RawlsGCN | This repository |
|---|---|---|
| Differentiate | Loss `f` (scalar) | Test node output `h_x^(k)` (vector) |
| w.r.t. | Weight parameters `W^(l)` (training effect) | Each node y's input features (prediction effect) |
| Scope | Global — node's role in training | Local — node y's role in a specific prediction |
| Stage | Training loop: node → gradient → weight update | Forward pass: node features → message passing → target representation |
| Closed form | Yes — `deg_Â(v)` factor | No — numerical Jacobian through trained weights |
| What it reveals | Degree causes systemic training unfairness (high-degree nodes dominate weight updates) | Which training nodes steer a specific test node's prediction, and how much |

**Critical distinction:** RawlsGCN's result is **structural** — the degree dependency holds regardless of the learned weights, because it comes from the renormalized adjacency matrix. This repository's influence is **empirical** — computed through the actual trained weights and nonlinearities, so two same-degree nodes can have very different influence distributions depending on what the model has learned. RawlsGCN explains *that* degree creates unfairness via a structural bound; this repository's Jacobian explains *how* that unfairness manifests for a specific node through the full computation path.

Both arrive at the same conclusion (degree causes unfairness) through complementary lenses: RawlsGCN from the training-time gradient perspective, this repository from the inference-time prediction perspective.

---

## Training Node Degree and Influence

### Does the Jacobian-L1 influence already account for degree normalisation?

Yes — the degree normalisation factors are baked into the Jacobian by construction.

For a 1-hop neighbour `y` in a 2-layer GCN, the derivative of focal node `x`'s output with respect to `y`'s input features is:

```
∂h_x^(2)[i] / ∂h_y^(0)[f]
  = ã_xy · Σ_j  σ'(z_x^(2)[i]) · W²[i,j] · ã_xx · σ'(z_x^(1)[j]) · W¹[j,f]
```

The factor `ã_xy = 1/sqrt((deg_x+1)(deg_y+1))` appears explicitly as a multiplicative scalar. The rest of the sum depends only on the focal node `x`'s pre-activations and the weight matrices — it is **the same for every 1-hop neighbour of x**. Therefore, among 1-hop neighbours of a fixed focal node, influence is proportional to `ã_xy`, which decreases with `deg_y`. Higher-degree 1-hop neighbours carry lower influence, directly from the Jacobian arithmetic.

For a 2-hop neighbour `z` routed through intermediate node `y`:

```
∂h_x^(2)[i] / ∂h_z^(0)[f]  ∝  ã_xy · ã_yz
  = 1/sqrt((deg_x+1)(deg_y+1))  ×  1/sqrt((deg_y+1)(deg_z+1))
  = 1 / ( sqrt(deg_x+1) · (deg_y+1) · sqrt(deg_z+1) )
```

The intermediate node `y`'s degree appears with a **full power of 1** in the denominator (not square root). A high-degree hub intermediate node therefore doubly attenuates the influence of 2-hop neighbours compared to a high-degree leaf node at 1-hop.

**Consequence for the "effective influence" measure.** An earlier version of `analyse_hop_influence.py` computed `eff_I_x[n] = I_x[n] × ã_xn` as a second table. This was incorrect: since `ã_xy` is already a factor inside `I_x[y]`, multiplying again double-counts the normalisation for 1-hop neighbours, and applies a hypothetical direct-edge weight for multi-hop neighbours that has no correspondence to anything in the actual computation. The effective influence table has been removed; the raw Jacobian-L1 table is the correct measure.

### Do high-degree nodes have higher influence in general?

It depends on the scope of the question — the answer differs for per-focal-node influence vs graph-wide influence.

**Per focal node (what `analyse_hop_influence.py` shows):** No. Among the neighbours of a fixed focal node `x`, a higher-degree neighbour `y` has *lower* influence because `ã_xy = 1/sqrt((deg_x+1)(deg_y+1))` shrinks as `deg_y` grows, and this factor scales the entire Jacobian for that node. The degree normalisation is a direct attenuation of high-degree neighbours.

**Graph-wide total influence:** Not necessarily lower. A node with degree `d` is a 1-hop neighbour of `d` different focal nodes. Its influence on each is attenuated by `1/sqrt(d+1)`, but summing across all `d` focal nodes gives total influence `∝ d / sqrt(d+1) ≈ sqrt(d)`, which **increases** with degree. High-degree hubs can therefore dominate the graph-wide influence budget even though their per-focal-node contribution is individually small.

For the per-hop influence table — which is always computed from the perspective of one focal node — the per-focal-node interpretation applies: higher-degree training neighbours exert less influence on the focal node.

### Is influence directly related to edge weight and hop distance?

Yes to both, and the two interact multiplicatively.

GCN normalises each edge (u → v) by `1 / sqrt(deg_u × deg_v)`. The Jacobian-based influence `I(x, y) = Σ |∂h_x^(k) / ∂h_y^(0)|` propagates through these normalisation coefficients as well as the learned weight matrices. For a 1-hop training node, influence scales roughly with its edge weight. For a 2-hop training node the coefficient compounds: the contribution passes through two edges, each with its own normalisation, so the effective structural weight is the product of both — and since each factor is < 1, the product is smaller still.

**Hop distance is a first-order effect.** The influence log for node 1362 makes this concrete:

| Node | Deg | Hop | Norm influence |
|------|-----|-----|---------------|
| 1597 (diff) | 8  | 1 | **0.6727** |
| 384  (diff) | 8  | 2 | 0.0110 |
| 271  (diff) | 78 | 1 | 0.0256 |
| 1371 (same) | 2  | 2 | 0.0088 |

Nodes 1597 and 384 have identical degree, but 1597 is at hop=1 while 384 is at hop=2 — their influence differs by ~60×. Hop distance dominates here. All four same-class training nodes are at hop=2; the dominant diff-class node (1597, deg=8) is at hop=1. The misclassification is therefore driven by both a degree advantage (deg=8 vs same-class degrees of 2–6) *and* a hop advantage (direct neighbor vs 2-hop). These two factors compound.

**Degree still matters within the same hop.** Node 271 (deg=78, hop=1) has only 2.6% influence despite being a direct neighbor — its very high degree reduces the edge weight to `1/sqrt(22×78) ≈ 0.024`, almost fully attenuating its contribution. Node 1597 (deg=8, hop=1) by contrast gets `1/sqrt(22×8) ≈ 0.075` — three times higher per edge, and it enters this as a factor in the full Jacobian.

**Some 2-hop nodes have unexpectedly high influence** (e.g. node 674: deg=5, hop=2, norm=0.1129). The path from 674 to 1362 passes through an intermediate node; if that intermediate node has low degree, both edge weights along the path are high, making the compounded product larger than expected. The intermediate node's degree is therefore also part of the influence budget — not just the endpoints.

The combined picture: influence is shaped by (1) hop distance, (2) the training node's own degree, and (3) the degrees of intermediate nodes along the path. Degree alone (as studied by the aggregation table) is insufficient; the full path structure determines the effective signal reaching the target node.

**Multiple low-degree paths compound influence — concrete evidence from node 2284.**

Node 2284 (degree=12, class=2, misclassified) has two 2-hop training nodes:

```
same_train node 829    deg=3   hop=2   norm=0.2598
diff_train node 1597   deg=8   hop=2   norm=0.7402
```

Node 1597 (deg=8) dominates node 829 (deg=3) at the same hop despite being higher-degree. Cross-referencing the aggregation tables for node 1597 and node 2284 reveals exactly why. Nodes 510 (deg=4), 887 (deg=5), and 1294 (deg=5) appear in **both** neighbourhoods — as diff-class neighbors of node 1597 (class=1) and as same-class neighbors of node 2284 (class=2), meaning they are class=2 nodes that sit between the two:

```
1597 → 510  (deg=4) → 2284
1597 → 887  (deg=5) → 2284
1597 → 1294 (deg=5) → 2284
```

Node 1597 reaches node 2284 through **three separate paths**, each passing through a low-degree intermediate (deg=4 or 5). Each path contributes independently to the Jacobian, and their contributions accumulate. Node 829, despite being lower-degree, likely reaches node 2284 through a single path with a higher-degree intermediate, suppressing its compounded weight.

This makes the **number of low-degree paths** between a training node and a target a key determinant of influence — potentially more important than the training node's own degree. A higher-degree training node with multiple low-resistance routes into a target's neighbourhood can dominate a lower-degree training node that has only one high-resistance route.

### Is it better to have lower-degree training nodes in the neighbourhood?

It depends on the class of the training node — and this is the core tension:

- **Lower-degree same-class training node:** high edge weight → strong, focused correct signal. Best case for the target node.
- **Lower-degree diff-class training node:** same high edge weight → strong, focused *incorrect* signal. Worst case, as seen with node 2248.

So low degree amplifies influence regardless of class. Whether that is beneficial depends entirely on which class the training node belongs to. A neighbourhood where same-class training nodes are low-degree and diff-class training nodes are high-degree would be optimal — the signal is concentrated on the right class. The reverse (low-degree diff-class, high-degree same-class) is the failure mode illustrated by node 387.

This suggests that the degree distribution of training nodes, split by class match, is a meaningful diagnostic: not just how many same/diff-class training nodes are in the receptive field, but what degree they have.

### Are lower-degree test nodes disadvantaged by having a high-degree same-class training node?

Yes. The edge weight `1 / sqrt(deg_u × deg_v)` penalises the training node's degree as well as the test node's. A low-degree test node (say degree=2) with a high-degree same-class training neighbour (say degree=50) receives a per-edge weight of `1 / sqrt(2 × 50) ≈ 0.10`. If instead the same-class training neighbour had degree=2, the weight would be `1 / sqrt(2 × 2) = 0.50` — five times stronger.

So a low-degree test node adjacent to a high-degree same-class training node gets a weaker direct signal from it than from a low-degree same-class training node. The training node's high degree dilutes the edge, even when the structural connection exists. This is a subtle form of the same degree-bias mechanism: not just whether a training node is nearby, but how much signal it can deliver per edge.

There is a potential counterpoint: high-degree training nodes aggregate from more neighbours during training and may therefore have richer, more stable learned representations. But the per-edge signal they deliver to test nodes is still attenuated by the normalisation, so the benefit of a better representation may not compensate for the lower edge weight at inference time.

### Reachability plots — two complementary views

The reachability analysis (`analyse_degree1_reachability.py --all-degrees`) produces two plots that answer different questions about training-signal access and misclassification.

**Plot 1 — "Why are misclassified nodes failing?" (current view)**
`reachability_by_degree.png` — stacked bar chart, one bar per degree group.

Each bar sums to 100% over the misclassified nodes at that degree. The three segments are:
- **Red**: fraction of misclassified nodes that had *no training node reachable at all* within k hops. These nodes had zero labelled signal to work from.
- **Orange**: fraction that had training nodes reachable, but *none of the same class*. Every labelled anchor was a wrong-class node — the model received supervision, but from the wrong source.
- **Blue**: fraction that had a same-class training node reachable, yet *still* failed. Something beyond reachability caused the error (degree-normalisation dilution, diff-class non-training neighbours, dead ReLU paths, etc.).

**Correct interpretation**: "X% of misclassified degree-D nodes had no training node reachable."
**Incorrect interpretation**: "X% of unreachable degree-D nodes are misclassified." — that is the flipped question, answered by Plot 2.

**Plot 2 — "Within each bucket, what fraction are correct vs misclassified?"**
`classification_split_by_bucket.png` — grouped bar chart, three bars per degree (one per bucket), each bar stacked into two segments.

For each degree on the x-axis, there are three side-by-side bars. Each bar represents one reachability bucket and shows what proportion of the nodes in that bucket at that degree are correctly classified (light shade, bottom) vs misclassified (dark shade, top). The bar height is always 100% — it is normalised within the bucket, not across buckets.

**Correct interpretation**: "Among degree-D nodes in the red bucket (no training reachable), Y% are misclassified and (100−Y)% are correctly classified."

Key things to look for: if the dark segment of the red bar is near 100% for low-degree nodes but shrinks at higher degrees, it suggests that reachability is a near-sufficient condition for failure only at low degrees — at higher degrees, nodes without reachable training signal may still be classified correctly on feature evidence alone. Conversely, if the blue bar (same-class train reachable) shows a substantial dark segment, it confirms that reachability of a same-class training node is necessary but not sufficient for correct classification.

### Degree-1 nodes — high misclassification driven by low reachability to training nodes

Degree-1 nodes tend to show a disproportionately high misclassification rate. The likely primary cause is **low reachability to training nodes** rather than the degree-normalisation imbalance that affects high-degree nodes.

**Why reachability is the bottleneck for degree-1 nodes:**

1. **Single neighbour limits structural access.** A degree-1 node has exactly one neighbour. For a 2-layer GCN (k=1 message-passing hop before the linear head), its receptive field is that one neighbour and any nodes two hops away via that neighbour. If the single neighbour is not a training node, direct supervision is absent — the node depends entirely on indirect signal propagated through an intermediary.

2. **Labelling ratio is low.** The probability that the single neighbour happens to be a training node is low — roughly equal to the global training node fraction (~4% for Cora with 20 labels per class). Most degree-1 nodes therefore have no direct training anchor at hop=1.

3. **High average SPL to training nodes.** With only one structural path into the graph, degree-1 nodes tend to have longer shortest-path distances to training nodes than higher-degree nodes, which have multiple paths and are more likely to sit near a densely-labelled hub.

4. **No redundancy in the signal path.** Higher-degree nodes can receive same-class signal through multiple neighbours even if some are diff-class — the correct signal still has other routes. A degree-1 node has no such redundancy. If its single neighbour is diff-class or poorly-trained, there is no alternative path.

**Contrast with the high-degree failure mode.** High-degree nodes fail because they aggregate too much — diff-class neighbours numerically overwhelm same-class signal through the degree-normalised sum. Degree-1 nodes fail because they aggregate too little — their single neighbour may not carry the right class signal at all, regardless of edge weights. These are opposite ends of the same structural spectrum.

**Connection to the node 1305 case study.** Node 1305 (degree=2) had a same-class training anchor (node 1555, deg=19) but was still misclassified because that anchor's high degree diluted its edge weight below that of the diff-class neighbour. A degree-1 node in the same situation would have only the one high-degree training anchor and no competing neighbours — yet if the anchor is high-degree, the effective signal is still weak. The degree-1 failure is therefore partly about the same edge-weight penalty, compounded by the complete absence of path redundancy.

---

## Open Questions / To Explore

### How do diff-class nodes dominate the aggregation numerically?

**To investigate.** The hypothesis is that for high-degree nodes, diff-class training nodes outnumber same-class ones in the k-hop neighbourhood, and GCN's degree-normalised aggregation amplifies this imbalance.
Possible angles:

- For a selected node, measure the **ratio of same-class to diff-class training nodes** as a function of degree — does this ratio drop as degree increases?
- Decompose the GCN aggregation: for each layer, track the fraction of the aggregated message mass that comes from diff-class vs same-class nodes (weighted by the degree-normalisation coefficients `1 / sqrt(deg_u * deg_v)`).
- Check whether the dominance is purely structural (neighbourhood class composition) or also weight-driven (the learned weight matrices amplify diff-class signals).

### Do diff-class training nodes have higher degree than same-class training nodes?

**To investigate.** If diff-class training nodes in a test node's neighbourhood tend to have higher degree than the same-class ones, they carry more aggregated signal into the network — both because GCN's normalisation coefficient `1 / sqrt(deg_u * deg_v)` is smaller for high-degree nodes (downweighting their direct contribution) but their embeddings are richer from aggregating more neighbours, and because high-degree training nodes influence a larger fraction of the graph through their own k-hop reach.

A secondary question: does higher degree in training nodes correlate with faster/better convergence during training? High-degree training nodes receive gradient updates that are an average over many neighbours, which may stabilise their gradients, while low-degree training nodes see noisier, sparser updates and may converge more slowly or to a worse local minimum. If same-class training nodes in the neighbourhood happen to be low-degree, their embeddings may be less well-trained, further reducing their effective influence on the target node's prediction.

### Learned weight suppression

**To investigate.** Beyond structural causes, the trained weight matrices themselves may systematically suppress same-class signal. Concretely: after training, do the weight matrices `W^(l)` project same-class neighbour features into directions that are near-orthogonal to the directions used for the final classification? This could be studied by:

- Inspecting the alignment between same-class neighbour embeddings (at each layer) and the learned class prototype vectors.
- Comparing the weight-matrix singular vectors against the class-discriminative directions in feature space.
- Checking whether the suppression is consistent across runs (i.e., a property of the optimisation landscape) or run-dependent.

### High-degree test node anomalies — broader study

**To investigate.** The influence analysis so far has been run on a single dataset (Cora) with the public split. The general trend (higher-degree nodes achieve higher accuracy) is established, but the specific anomalous misclassifications motivate a broader study. To determine whether these anomalies are general:

- **Across training splits:** repeat the influence analysis across multiple random splits and public splits — do the same high-degree test nodes consistently show near-zero same-class influence, or does it vary with which nodes are labelled?
- **Across datasets:** run on CiteSeer, PubMed, ogbn-arxiv, and heterophilic graphs (e.g. Chameleon, Squirrel) — does the diff-class dominance pattern hold, and is it stronger on homophilic vs heterophilic graphs?
- **Across models:** compare GCN vs GCNII vs GAT — does attention help high-degree nodes recover same-class signal, or does the same suppression appear regardless of architecture?

---

*Last updated: 2026-04-16*
