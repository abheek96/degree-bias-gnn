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

**Contrast with node 1362.** Node 1362 (degree=22) had `same_train=4`, `diff_train=11` — many training nodes of both classes in-neighbourhood, but same-class influence ≈ 0 due to dead ReLU paths / learned weight suppression. Node 387 is simpler structurally: it has one same-class training anchor that is correctly active, but a single low-degree wrong-class node overwhelms it purely through degree-normalisation arithmetic. These are two distinct mechanisms by which diff-class influence can dominate.

### Does `same_class_influence ≈ 0` mean those training nodes weren't trained?

No. All training nodes participate in the loss and gradient updates as usual.
The influence score is a property of the **forward pass of the trained model**, not of the training procedure.
It measures how much the trained model *uses* information from each neighbour when making a prediction — not whether that neighbour was part of training.
A training node with zero influence simply means the model has learned (or failed to learn) weights such that this node's features do not percolate meaningfully to the target node's representation.

---

## Training Node Degree and Influence

### Is influence directly related to edge weight?

Yes, structurally. GCN normalises each edge (u → v) by `1 / sqrt(deg_u × deg_v)`. The Jacobian-based influence `I(x, y) = Σ |∂h_x^(k) / ∂h_y^(0)|` propagates through these normalisation coefficients as well as the learned weight matrices. For a 1-hop training node, influence scales roughly with its edge weight. For a 2-hop training node, the coefficient compounds: the contribution passes through two edges, each with its own normalisation, so the effective weight is the product of both. This is why node 2248 (degree=2) at 2 hops in the node 387 case still dominated 85% of influence — two high edge-weight hops multiplied together. The learned weights modulate this further, but the structural degree-normalisation is the primary driver of the disparity when training node degrees differ substantially.

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

*Last updated: 2026-03-22*
