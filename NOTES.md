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

### Does `same_class_influence ≈ 0` mean those training nodes weren't trained?

No. All training nodes participate in the loss and gradient updates as usual.
The influence score is a property of the **forward pass of the trained model**, not of the training procedure.
It measures how much the trained model *uses* information from each neighbour when making a prediction — not whether that neighbour was part of training.
A training node with zero influence simply means the model has learned (or failed to learn) weights such that this node's features do not percolate meaningfully to the target node's representation.

---

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

*Last updated: 2026-03-17*
