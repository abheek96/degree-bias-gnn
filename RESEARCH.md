# Degree Bias in Graph Neural Networks — Research Documentation

> Last updated: 2026-03-17

---

## 1. Research Problem

Graph Neural Networks (GNNs) for node classification are known to perform unevenly across nodes, but the structural reasons for this are not well understood. This project investigates **degree bias**: the hypothesis that a node's connectivity (its degree in the graph) systematically affects how well a GNN can classify it.

The investigation has revealed that **higher-degree nodes generally achieve higher accuracy** — consistent with the intuition that more connections bring richer aggregated signal. However, this is not always the case: certain mid- and high-degree nodes are anomalously misclassified despite the general trend. This motivates investigation into other graph and model properties to explain why degree alone does not fully determine performance.

### 1.1 How this work goes beyond prior literature

Existing work on degree bias in GNNs has largely studied the problem on the **standard public splits** of benchmark datasets. This repository extends that baseline in three ways:

1. **Multiple split regimes.** Beyond the public split, the analysis is also run on **random splits** — where the 20 training nodes per class are drawn randomly rather than fixed. This tests whether the degree-bias findings hold only for the specific nodes labelled in the public split, or whether they are a robust property of the graph structure regardless of which nodes happen to be labelled.

2. **Connected component filtering.** The analysis is restricted to the **largest connected component** of the graph. Small disconnected components, by construction, cannot have any training nodes in their local neighbourhood. Misclassifications in those components would conflate degree bias with a trivially different problem (complete absence of training signal) and contaminate the results. By working on the CC only, every node in the analysis is reachable from training nodes via message passing.

3. **Granular, degree-wise accuracy analysis.** Prior work typically splits nodes into broad groups (e.g. "low-degree" vs "high-degree" based on a fixed threshold). This repository instead examines accuracy **at each individual degree value**, producing fine-grained distributions. This granular view reveals anomalous misclassification at specific degree values that would be masked by coarse grouping — and directly refutes the implicit assumption that nodes with similar degree should have similar performance. The anomalies motivate looking beyond degree alone.

The central questions are:

- What is the general relationship between node degree and classification accuracy, and where does it break down?
- What structural properties of a node's neighbourhood explain anomalous misclassification among specific mid/high-degree nodes?
- How does model depth (number of layers) interact with degree to affect accuracy?
- Does the training signal that reaches a test node (via message passing) differ in quality for anomalously misclassified nodes compared to correctly classified ones of similar degree?
- Can the trained model's internal influence distribution explain misclassification at certain degree groups?

---

## 2. Task Setup

### 2.1 Dataset

**Cora** — a standard citation network benchmark.

| Property | Value |
|---|---|
| Nodes | ~2,708 |
| Edges | ~5,429 (undirected) |
| Node features | 1,433-dimensional bag-of-words |
| Classes | 7 (paper topics) |
| Task | Semi-supervised node classification |

### 2.2 Connected Component (CC) filtering

**Setting used: `use_cc: true`**

Cora as distributed contains a handful of isolated nodes and small disconnected components. These are dropped, keeping only the largest connected component.

**Why:** GNNs propagate information through edges. Nodes in disconnected components can never receive any message-passing signal from the training nodes in the main component. Including them would artificially inflate the "unreachable node" problem and confound the degree-bias analysis with a connectivity-bias that has a trivial cause — a node is misclassified not because of degree-related aggregation dynamics, but simply because no training signal can reach it at all. The CC flag ensures every node in the analysis can, in principle, receive training signal, making misclassification attributable to structural and aggregation factors rather than complete disconnection. This is a methodological choice not made consistently in the prior literature on degree bias.

### 2.3 Data splits

**Setting used: `split: public`**

| Split | Size |
|---|---|
| Train | 20 nodes per class = 140 total |
| Val | 500 nodes |
| Test | 1,000 nodes |

**Why public split (and why random splits matter):** The standard Planetoid public split for Cora uses a fixed, widely-used assignment of 20 training nodes per class. Prior work on degree bias has largely relied on this split alone. Starting from the public split here allows direct comparison to the existing literature.

However, a public-split-only analysis raises the question of whether degree-bias findings are properties of the graph structure or of the particular nodes that happen to be labelled. This repository therefore also supports **random splits** (`split: random`) where training nodes are drawn uniformly at random (same per-class count). Key reasons:

- Multiple runs (5 seeds) across random splits quantify how sensitive the degree-bias findings are to *which* nodes are labelled, not just to model initialisation.
- Comparing public vs random split results tests whether the anomalous misclassifications persist across different training node placements or are an artefact of one fixed labelling.
- It enables studying how training node degree distribution (are high-degree nodes in the training set or not?) interacts with test-node degree bias.

**Seed:** base seed 42; runs use seeds 42, 43, 44, 45, 46. The split itself is fixed across all runs within one execution (set before the run loop), so only model initialisation varies between runs.

### 2.4 Hyperparameter configurations

#### Base configuration (`config.yaml`)

| Parameter | Value |
|---|---|
| Hidden dim | 512 |
| Layers | 2 |
| Dropout | 0.0 |
| LR | 0.001 |
| Weight decay | 5e-4 |
| Epochs | 200 |
| Early stopping patience | 50 |
| Runs | 5 |

#### GCN on Cora (`configs/GCN_Cora.yaml`)

Same as base. Layer sweep for `acc_vs_degree_by_layers`: [1, 2, 3, 4, 5].

#### GCNII on Cora (`configs/GCNII_Cora.yaml`)

| Parameter | Value |
|---|---|
| Hidden dim | 64 |
| Layers | 64 |
| Dropout | 0.5 |
| LR | 0.01 |
| Patience | 100 |
| Layer sweep | [16, 32, 64] |

GCNII (Chen et al., 2020) uses initial residual connections and identity mapping to avoid over-smoothing, which allows it to be trained with many more layers than a standard GCN. Including it tests whether architectural choices that mitigate over-smoothing also mitigate degree bias.

---

## 3. Models

### GCN (Kipf & Welling, 2017)
Standard graph convolutional network. Each layer computes:

```
H^(l+1) = σ( D̃^{-1/2} Ã D̃^{-1/2} H^(l) W^(l) )
```

where `Ã = A + I` (adjacency with self-loops) and `D̃` is the corresponding degree matrix. The `1/sqrt(deg_u * deg_v)` normalisation is the key aggregation mechanism — high-degree nodes average over more neighbours, potentially diluting the signal from any individual neighbour.

The architecture has `num_layers - 1` GCNConv layers followed by one `nn.Linear(hidden_dim, out_dim)` classification head. Setting `num_layers=3` gives 2 graph convolution layers and 1 linear layer.

### GCNII (Chen et al., 2020)
Extends GCN with two modifications per layer:
1. **Initial residual:** each layer's output is a convex combination with the initial embedding `H^(0)`.
2. **Identity mapping:** a scaled identity matrix is added to the weight matrix.

This allows very deep networks (64+ layers) without over-smoothing. Used here to compare degree-bias behaviour as a function of depth.

---

## 4. Metrics

All metrics are computed on **test nodes** unless stated otherwise.

### 4.1 Accuracy by degree (`acc_vs_degree`)

For each unique test-node degree, the classification accuracy is computed per run and the distribution across runs is reported as a boxplot.

**Why:** The primary metric for characterising degree bias. The general trend — whether accuracy increases, decreases, or is flat with degree — establishes the baseline. Deviations from this trend at specific degree values (nodes that are anomalously worse or better than their degree group) are the focal point: they indicate that degree alone does not determine performance and that other structural or model properties are at play.

### 4.2 k-hop degree (`acc_vs_khop_degree`)

The k-hop degree of a node is the number of distinct nodes reachable within at most k hops (k = `khop_k`, default 2). This equals the size of the node's receptive field for a k-layer GNN.

**Why:** 1-hop degree measures direct connectivity, but a GNN's actual receptive field is its k-hop neighbourhood. Two nodes with the same 1-hop degree can have very different 2-hop degrees (and thus very different amounts of aggregated information). Studying accuracy vs. k-hop degree separates structural richness from raw connectivity.

### 4.3 Distance to training nodes (`acc_vs_distance`)

Two hop-distances per test node:

- **dist_to_train:** minimum number of hops to reach *any* training node.
- **dist_to_same_class_train:** minimum number of hops to reach a training node of the *same class*.

Grouped by test-node degree and plotted as median + IQR boxplots.

**Why:** A GNN can only propagate information from labelled nodes through edges. A test node that is 4 hops from the nearest same-class training node receives heavily diluted and potentially corrupted label signal. This metric links structural distance to training data with the degree of the test node.

### 4.4 Average Shortest Path Length (SPL) (`spl_vs_degree`)

Two variants:

- **SPL to all training nodes:** average BFS distance from a test node to every training node that can reach it.
- **SPL to same-class training nodes:** same, restricted to training nodes sharing the test node's true label.

Computed for all test nodes and plotted degree-wise as boxplots.

**Why:** Distance to the nearest training node is a binary threshold. Average SPL gives a richer picture — a test node might have one nearby training node but be far from all others, meaning it receives strong signal from only a tiny fraction of the training set. If high-degree nodes also tend to have higher average SPL to same-class training nodes, that is a compounding disadvantage.

### 4.5 Labelling ratio (`labelling_ratio`)

A binary indicator per test node: does it have at least one direct (1-hop) training node as a neighbour?

Plotted both as a standalone degree-wise line and overlaid with accuracy (twin-axis) to show the co-variation.

**Why:** A test node with a labelled immediate neighbour receives direct, un-diluted label signal in layer 1 of the GNN. Nodes without any labelled neighbour must rely entirely on indirect signal propagated through non-training intermediaries. The labelling ratio captures this critical first-hop availability.

### 4.6 Neighbourhood purity (`purity_vs_degree`)

For a node v at expansion radius k:

```
purity_k(v) = |{u ∈ N_k(v) : label[u] == label[v]}| / |N_k(v)|
```

Computed for **all nodes** (not just test nodes) for k = 1 to `purity_k_max` (default 2).

**Why:** Even if a test node has training neighbours, those neighbours may be of a different class, injecting misleading signal. Purity measures the class composition of the neighbourhood. Low purity = high class heterogeneity = more noise in the aggregated message.

**Delta purity** (`plot_purity_delta_by_degree`): plots `purity(k_max) - purity(k_min)` per degree group. A negative delta means that expanding the neighbourhood reduces class homogeneity — the signal gets noisier as more layers are added. This is expected for high-degree nodes in a heterophilic portion of the graph.

### 4.7 Influence scores (`influence_analysis`)

For a selected test node x (chosen by degree from `influence_degrees` in config), the exact influence distribution is computed:

```
I(x, y)  = Σ_{i,f}  |∂h_x^(k)[i] / ∂h_y^(0)[f]|
I_x(y)   = I(x, y) / Σ_z I(x, z)
```

This is computed via an exact Jacobian (`torch.autograd.functional.jacobian`). The influence scores of all training nodes within the k-hop receptive field are then split into same-class and diff-class groups.

**Why:** Accuracy and structural metrics describe *what* happens to a node but not *why* the model fails. The influence distribution directly asks: which input nodes is the trained model actually using to form its prediction for node x? If a misclassified high-degree node receives near-zero influence from its same-class training neighbours and high influence from diff-class neighbours, that is causal evidence of the aggregation mechanism failing for that node.

### 4.8 Training-neighbor degree distribution (`train_neighbor_degree`)

For each test node, finds all training nodes in its k-hop receptive field and records the **mean degree** of same-class vs diff-class training neighbors. Reports per-test-node:

- `same_mean_deg`: mean degree of same-class training nodes in the k-hop field
- `diff_mean_deg`: mean degree of diff-class training nodes in the k-hop field
- `same_count` / `diff_count`: number of training nodes in each group

The "degree advantage" is defined as `same_mean_deg − diff_mean_deg`.

**Why:** In GCN, the aggregation weight for edge (u→v) is `1/sqrt(deg_u * deg_v)`. A same-class training node with high degree contributes *less* per edge than a low-degree one. If diff-class training nodes systematically have lower degree than same-class ones, their per-edge signal is stronger despite being fewer in number, potentially driving misclassification. This metric separates the *count* and *degree* components of the aggregation imbalance — two nodes with the same count but different degree profiles contribute very differently to the aggregated message.

---

## 5. Plots

### 5.1 `acc_vs_degree`

**What:** Boxplots of per-degree classification accuracy across 5 runs, with a secondary panel showing the number of test nodes at each degree.

**Why:** Primary visualisation of degree bias. Unlike prior literature which typically bins nodes into broad "low-degree" and "high-degree" groups based on a fixed threshold, this plot examines accuracy at **each individual degree value**. The granular view is essential: it reveals anomalous misclassification at specific degree values (e.g. a node of degree 22 failing while degree-20 and degree-24 nodes succeed) that coarse grouping would average away. This directly challenges the assumption that nodes with similar degree should perform similarly, and is what motivates looking beyond degree as the sole explanatory factor. The node count panel is critical — conclusions about high-degree groups are only meaningful if there are enough nodes to support them.

---

### 5.2 `acc_vs_distance` (combined plot)

**What:** For each degree group, two boxplot distributions shown side by side: dist_to_train (any class) and dist_to_same_class_train. Accuracy overlay on a twin axis.

**Why:** Investigates whether structural proximity to training data differs for anomalously misclassified nodes compared to well-classified nodes of similar degree. The combination of the two distance types reveals whether the problem is (a) no labelled node nearby at all, or (b) plenty of labelled nodes but the same-class ones are far away — which would indicate class-specific signal quality as a factor beyond raw degree.

---

### 5.3 `acc_vs_khop_degree`

**What:** Same format as `acc_vs_degree` but grouped by 2-hop degree instead of 1-hop degree.

**Why:** The k-hop degree is the actual size of the GNN's receptive field. **The idea behind behind this is to evaluate how the model performs on nodes whose receptive fields has been increase by k+1, i.e, from original from 1-hop to 2-hop in this case.**  A node with 1-hop degree 3 but 2-hop degree 300 is in a densely connected region and behaves differently to a node with 1-hop degree 3 and 2-hop degree 5. This separates "locally sparse, globally dense" nodes from "locally sparse, globally sparse" ones.

---

### 5.4 `acc_vs_degree_by_layers`

**What:** For each (model, number of layers) combination, the mean accuracy per degree group is shown as one line. A secondary "trend" plot shows the slope of accuracy vs. layer count per degree group.

**Why:** **To test if adding more layers (higher k-hops) would improve the performance of low-degree nodes without compromising the performance of the higher degree ones.** Over-smoothing predicts that more layers should hurt high-degree nodes more (their neighbourhoods grow faster, averaging over more heterogeneous signal). This plot tests whether that prediction holds and whether GCNII mitigates it.

---

### 5.5 `spl_vs_degree` (two variants)

**What:** Boxplots of average SPL per degree group. Produced once for SPL to all training nodes, once for SPL to same-class training nodes only.

**Why:** Distinguishes two different failure modes:
- High avg SPL to *all* train → test node is in a structurally remote part of the graph.
- High avg SPL to *same-class* train only → same-class training nodes are scarce or clustered far away, even though other-class training nodes may be nearby.

The second variant is more diagnostic for bias because it isolates the class-specific distance.

---

### 5.6 `labelling_ratio_vs_degree` and `acc_and_labelling_ratio_vs_degree`

**What:** (1) Fraction of test nodes per degree group that have at least one direct training neighbour. (2) Median accuracy (blue) and labelling ratio (orange) on the same plot with a twin y-axis.

**Why:** Shows how the availability of directly labelled neighbours varies with degree and whether it co-varies with accuracy. Nodes without any labelled direct neighbour must rely on indirect signal — this metric can help explain anomalous misclassifications at specific degree groups where labelling ratio drops.

---

### 5.7 `purity_vs_degree` and `purity_delta_by_degree`

**What:** (1) Mean purity per degree group for each k value (1 to `purity_k_max`). (2) Delta purity = `purity(k_max) - purity(k_1)` per degree group, summarising how purity degrades as the neighbourhood expands.

**Why:** High-degree nodes are more likely to have heterogeneous neighbourhoods (in a graph where classes are not perfectly clustered). As more layers are added, the effective neighbourhood grows, pulling in more cross-class signal. The delta purity plot quantifies exactly how much the neighbourhood quality degrades per degree group — directly predicting which nodes should suffer most from additional GNN layers.

---

### 5.8 `influence_analysis` (aggregate bar plot)

**What:** For the selected test node(s), two side-by-side bars show the total normalised influence attributable to same-class training nodes (blue) vs diff-class training nodes (orange) within the k-hop receptive field. A secondary line shows the fraction of influence from diff-class nodes. Numeric labels are shown above each bar so small values remain readable.

**Why:** Converts the structural observations into a model-behaviour measurement. The bars directly show how the trained model distributes its attention between helpful (same-class) and potentially harmful (diff-class) training signal sources.

---

### 5.9 `influence_per_neighbor` (per-node bar chart)

**What:** One figure per selected node. Each bar is one training node in the k-hop receptive field, coloured blue (same-class) or orange (diff-class). Sorted left-to-right by descending influence score. X-axis labels are node indices.

**Why:** The aggregate bars show the totals but hide the distribution within each group. This plot reveals whether diff-class influence is dominated by one or two highly influential nodes or spread evenly across many, and whether same-class nodes are all near-zero or if one same-class node has some influence while others are zero.

---

### 5.10 `train_neighbor_degree` (two-panel degree comparison)

**What:** Two stacked panels per run.

- **Top panel:** For each test-node degree group, side-by-side boxplots of the mean degree of same-class training neighbors (blue) and diff-class training neighbors (orange) within the k-hop receptive field.
- **Bottom panel:** For each test-node degree group, the "degree advantage" (mean_deg_same − mean_deg_diff) shown as side-by-side boxplots for correctly classified (green) vs misclassified (red) nodes. A zero line marks parity. Positive values mean same-class training nodes have higher average degree; negative means diff-class nodes do.

**Why:** Separates two distinct components of the aggregation imbalance that prior metrics conflate. The count-based imbalance (same_count vs diff_count) tells us *how many* training nodes of each class are in the receptive field. The degree-based metric tells us *how strongly* each contributes per edge via the GCN normalisation factor `1/sqrt(deg_u * deg_v)`. A node could have 4 same-class neighbors and 11 diff-class neighbors (count disadvantage), but if the same-class nodes are degree-1 leaf nodes and the diff-class nodes are degree-20 hubs, the effective aggregation weight favors the same-class signal. Conversely, if same-class nodes are the hubs, their signal is more diluted. This plot directly tests whether misclassified nodes share a consistent pattern in the degree structure of their training neighborhood.

---

## 6. Observations So Far

> Based on initial runs of GCN (2 layers, hidden dim 512) on Cora with public split, 5 seeds.

### Accuracy vs degree
- Accuracy is **not monotone in degree**. Low-degree nodes (degree 1–2) tend to have high variance — some are well-classified, some are not, driven by whether their single neighbour happens to be a same-class training node.
- Mid-degree nodes show the most stable accuracy across runs.
- **The general trend is that higher-degree nodes achieve higher median accuracy** — more connections bring richer aggregated signal. However, certain specific mid/high-degree nodes are anomalously misclassified despite this trend. This is the central observation motivating the investigation: degree alone is not the deciding factor, and understanding what distinguishes these anomalous failures from the majority of well-classified high-degree nodes requires examining other structural and model properties.

### Distance to training nodes
- High-degree test nodes do not necessarily have larger dist_to_train — in a dense graph, high-degree nodes often have shorter paths to training nodes by virtue of having more edges.
- However, dist_to_same_class_train is more variable and can be larger for high-degree nodes because more edges also means more routes to *wrong-class* training nodes, increasing the chance that the nearest same-class training node is reached only via a longer, indirect path.

### Labelling ratio
- The fraction of test nodes with at least one direct training neighbour drops for very low and very high degree groups.
- For low-degree nodes the reason is obvious: fewer edges means fewer chances to be adjacent to a training node.
- For high-degree nodes the reason is dilution: even if one training neighbour exists, its signal is averaged with many non-training neighbours.

### Neighbourhood purity
- Purity decreases as degree increases: high-degree nodes have more heterogeneous neighbourhoods.
- Delta purity (purity at k=2 minus purity at k=1) is negative across all degree groups, confirming that expanding the neighbourhood always introduces more cross-class signal.
- The degradation is stronger for high-degree nodes — their 2-hop neighbourhood explodes in size, sampling a larger and more diverse slice of the graph.

### Influence analysis (node 1362, degree 22, GCN 2-layer)
- **same_train=4, diff_train=11** in the 2-hop receptive field.
- **same_class_influence ≈ 0 (order 1e-7), diff_class_influence ≈ 0.05.**
- Despite 4 same-class training nodes being structurally present in the receptive field, the trained model routes essentially zero gradient influence through them.
- The most likely causes: dead ReLU paths suppressing gradient flow through same-class neighbours; degree-normalisation diluting each individual neighbour's contribution; diff-class nodes numerically dominating the aggregation due to their greater count (11 vs 4).
- This is the clearest evidence so far that the model does not just *see* biased signal — it has *learned* to route around same-class signal for this node, amplifying the structural imbalance.

---

## 7. The Emerging Story of Degree Bias

### 7.1 Narrative: how the evidence accumulates

The investigation started with a simple empirical question: does degree predict accuracy? The answer is not a clean "yes" or "no" — it is a more nuanced story that unfolds in layers.

**Layer 1 — The accuracy signal is real but not monotone.**
Low-degree nodes are not consistently worse; they are *unstable*. A node with degree 1 has exactly one neighbour. If that neighbour is a same-class training node, the node is almost certainly classified correctly. If it is not, the node has almost no signal at all. The outcome is effectively a coin flip conditioned on the specific split. High-degree nodes, by contrast, tend toward *higher* median accuracy as degree increases — more connections provide richer aggregated signal on average. Yet this general trend conceals a crucial anomaly: specific mid/high-degree nodes are persistently misclassified. The signal is that **high degree is generally beneficial, but it does not guarantee correct classification**, and the exceptions need explanation.

**Layer 2 — The structural reason: neighbourhood quality degrades with degree.**
Purity analysis confirms that high-degree nodes have more heterogeneous neighbourhoods. On Cora — a citation network where papers cite across topic boundaries — nodes that are heavily connected are hubs that span multiple research communities. Their k-hop neighbourhood therefore samples a wider cross-section of classes. As GNN layers stack, the effective neighbourhood (the receptive field) grows, and the class signal becomes progressively noisier. The delta-purity plots show this degradation is steeper for high-degree nodes than for low-degree ones. For most high-degree nodes, richer aggregation still outweighs the noise. But for those anomalous nodes where the same-class training signal is particularly sparse or diluted, the noise can tip the balance toward misclassification.

**Layer 3 — The training signal story: same-class signal is sparse and far away for the anomalous cases.**
The SPL and distance metrics reveal a subtler structural disadvantage. High-degree test nodes are not necessarily far from training nodes in terms of raw hop-distance — they often have short paths to labelled nodes because they have many edges. But many of those nearby training nodes are of a *different class*. For anomalously misclassified high-degree nodes, the distance to the nearest same-class training node is larger and more variable. The first same-class label signal that reaches such a node through message passing has already been diluted and mixed with cross-class signal from the intermediate nodes along the path. Well-classified high-degree nodes, by contrast, tend to have same-class training nodes closer or more numerous in their neighbourhood.

**Layer 4 — The model internalises and amplifies the imbalance.**
The influence analysis delivers the most striking finding: for a degree-22 misclassified test node with 4 same-class and 11 diff-class training nodes within its 2-hop receptive field, the trained model assigns near-zero influence (order 1e-7) to the same-class training nodes and non-trivial influence (~0.05) to the diff-class ones. This is not just a structural observation — the model has *learned* to essentially ignore the same-class training signal available to it. The structural imbalance (more diff-class neighbours) is present at the input, but the learned weight matrices appear to reinforce rather than correct it.

---

### 7.2 Most probable explanation for anomalous misclassifications

Based on all evidence gathered so far, the most probable explanation for why *certain specific* high-degree nodes are misclassified despite the general trend is a **compounding cascade of four factors**:

1. **Neighbourhood class imbalance (structural).** High-degree nodes in Cora sit at the intersection of multiple research communities. For those anomalous nodes, the local neighbourhood happens to have significantly more diff-class training neighbours than same-class ones. Even with 20 training nodes per class, this imbalance is not unusual for hub nodes. The training signal available in the immediate neighbourhood is structurally skewed against correct classification for these specific nodes.

2. **Degree-normalised aggregation dilutes individual signals (architectural).** GCN normalises each neighbour's contribution by `1/sqrt(deg_u * deg_v)`. For a degree-22 node, every neighbour's message is scaled down by at least `1/sqrt(22) ≈ 0.21`. With 15 training-node neighbours, none of them individually dominates the aggregation — but diff-class nodes win by sheer count (11 vs 4). There is no mechanism in standard GCN to upweight same-class signal. Most high-degree nodes avoid this problem because their same-class training neighbours are more numerous or closer; the anomalous ones are the unlucky cases where the count imbalance is pronounced.

3. **Learned weight suppression reinforces the input bias (optimisation).** The model is trained to minimise cross-entropy over the training nodes, not the test nodes. If the global loss landscape is better served by weight matrices that process the dominant (diff-class-heavy) signal in a certain way, the weights converge accordingly. The result is that same-class training nodes in the neighbourhood of anomalous high-degree test nodes end up in a low-gradient region — the model has learned not to use them, not because they were absent, but because using them was not consistently rewarded during training given the broader dataset statistics.

4. **Propagation distance amplifies noise (depth).** Even at 2 layers, the same-class training signal for an anomalous high-degree test node must often travel through intermediate non-training nodes before reaching the target. Each hop introduces averaging over that intermediate node's full neighbourhood, further diluting the class-specific component. For nodes whose nearest same-class training node is 2 hops away (rather than 1), this two-step dilution is enough to make the same-class signal computationally negligible relative to the noisier but volumetrically larger diff-class signal.

**In short:** it is not one mechanism but the combination of all four — structural neighbourhood imbalance, architectural aggregation normalisation, learned weight suppression, and propagation dilution — that explains why *certain* high-degree nodes are anomalously misclassified. Most high-degree nodes avoid this fate because their neighbourhood class composition happens to be more favourable. The anomalous cases are those where all four factors compound: biased neighbourhood structure → biased training signal → biased weights → misclassification despite the node's structural richness.

---

### 7.3 Is training-signal neighbourhood analysis sufficient to explain the anomalies?

**Partially, but not completely.**

The neighbourhood training-signal analysis (SPL, labelling ratio, purity, influence) captures the *proximal* cause for specific misclassification events: the signal arriving at the anomalous node during inference is dominated by diff-class information. This explains individual events like node 1362.

However, there are aspects it does not yet explain:

- **What distinguishes anomalous misclassified high-degree nodes from the majority that are correctly classified.** Most high-degree nodes succeed despite having heterogeneous neighbourhoods — the analysis needs to identify the specific combination of factors (local purity, training node placement, feature-space properties, neighbourhood class counts) that tips an otherwise-advantaged high-degree node into misclassification.

- **The role of node features.** The influence analysis uses a Jacobian over the *input features* `h^(0) = x`. If the node's own features are highly class-discriminative, the model may classify correctly even with poor neighbourhood signal. The current analysis does not separate feature-driven classification from neighbourhood-driven classification.

- **Cross-run consistency.** A finding observed at one node in one run may not generalise. Whether the same nodes are consistently misclassified across seeds (and whether the influence pattern is consistent) has not yet been verified.

- **Whether the model *could* correct for the bias but doesn't, or structurally cannot.** The learned-weight-suppression hypothesis suggests the model *learns* to ignore same-class nodes. But it is possible that for some configurations, the same-class signal is too weak to be recovered even with ideal weights. Distinguishing these two scenarios requires intervention experiments (e.g., manually upweighting same-class gradients) that have not been run.

---

### 7.4 Should higher hops and larger receptive fields be investigated?

**Yes, and for specific reasons — but with important caveats.**

#### Why it is worth doing

At 2 layers (the current setting), the receptive field of node 1362 is its 2-hop neighbourhood. The influence analysis shows that 4 same-class training nodes exist within this field but carry near-zero influence. A natural question is: what happens at 3 or 4 layers? There are two competing hypotheses:

- **More layers help:** a 3-layer GCN can see 3-hop neighbours. If same-class training nodes that are currently 3 hops away carry stronger signal than the ones at 2 hops (e.g., because the 3-hop same-class nodes are in a more homophilic local region), adding a layer could recover same-class signal and improve classification of high-degree nodes.

- **More layers hurt:** a 3-layer receptive field is even larger. For a degree-22 node, the 3-hop neighbourhood could contain hundreds of nodes. More nodes means more averaging, more class mixing, and potentially worse purity. The diff-class training nodes that already dominate at 2 hops will continue to dominate at 3 hops, and new diff-class training nodes at 3 hops may join them, worsening the imbalance.

The purity analysis already provides a structural prediction: delta purity is negative and steeper for high-degree nodes, suggesting more layers should hurt. But the influence analysis would confirm whether this structural prediction translates into the model's actual gradient routing behaviour.

Studying this systematically — running the influence analysis at k=3 and k=4 for the same selected nodes — would directly test whether the near-zero same-class influence is a property of the specific 2-hop neighbourhood or a deeper feature of how the graph is structured around high-degree nodes at all scales.

#### Why to be cautious

- **Computational cost.** The influence analysis uses an exact Jacobian (`torch.autograd.functional.jacobian`), which requires one backward pass per output dimension. At 2 layers with 512 hidden dim and 7 output classes, this is already expensive. At k=4 layers, the receptive field for a degree-22 node could contain thousands of nodes. The Jacobian still has the same shape `[7, N, 1433]` regardless of k, but the model is deeper and the computation is heavier. This is feasible for a handful of selected nodes but not for a sweep.

- **Over-smoothing conflation.** As layers increase, the model also becomes susceptible to over-smoothing (all node embeddings converging to similar values), which is a separate failure mode from degree bias. Influence scores at k=4 might be near-zero for *all* neighbours — same and diff class alike — not because of degree bias but because the model has over-smoothed. The two effects must be disentangled, which requires comparing against a model that does not over-smooth (e.g., GCNII at the same depth).

- **Selecting the right nodes.** At higher k, the receptive field grows so large that the concept of "same-class training nodes within the field" becomes almost trivially true — almost all training nodes are eventually reachable. The interesting regime is specifically where same-class training nodes exist in the field but carry low influence despite being reachable. This selection needs to be preserved carefully as k increases.

**Recommendation:** run the influence analysis for the same selected high-degree nodes across k = 2, 3, 4 layers, using both GCN and GCNII side by side. For GCN, expect the same-class influence to remain near zero or decrease further as k grows (over-smoothing + purity degradation compounding). For GCNII, the initial residual connection may preserve some same-class signal even at greater depth, providing a contrastive baseline. If GCNII shows higher same-class influence at k=4 than GCN does at k=2, that is strong evidence that the architectural choice — not just the structural neighbourhood — determines whether the model can recover the same-class signal that is structurally present.

---

### 7.5 What the plots actually establish — and what they don't

It is important to be precise about the evidentiary status of each metric, because the structural metrics and the model-behaviour metrics are answering fundamentally different questions.

#### What the plots structurally confirm

**Purity vs degree** gives the clearest and most direct structural trend. Purity decreases as degree increases, and the delta purity (k=2 minus k=1) is negative and steeper for high-degree nodes. This directly says: the neighbourhood of a high-degree node is more class-heterogeneous, and the heterogeneity worsens as the receptive field expands. This is a clean, graph-structural explanation for *why* high-degree nodes should be harder to classify — the input signal they receive is noisier by construction.

**SPL to same-class training nodes** provides a complementary structural view. If high-degree nodes have longer average paths to same-class training nodes (even when their raw dist_to_train is short), that confirms the same-class label signal arrives more diluted and mixed with cross-class signal from intermediate nodes.

**Labelling ratio overlaid with accuracy** shows whether having a direct labelled neighbour correlates with correct classification. A strong co-variation confirms that first-hop label availability is a meaningful driver of accuracy — and its absence disproportionately affects certain degree groups.

Together, these three metrics tell a consistent and plausible structural story about the anomalous cases: while high-degree nodes generally benefit from richer aggregation, those that are anomalously misclassified tend to have noisier neighbourhoods, are farther from same-class label sources, and are less likely to have a directly labelled same-class neighbour whose signal arrives undiluted.

#### What the plots have not yet established

The critical gap is that **all of the above are structural properties of the graph, not observations about what the trained model actually does**. Purity does not inspect the model — it describes the neighbourhood composition. SPL does not inspect the model either — it describes the graph topology. These metrics explain why the *input signal available* to high-degree nodes is structurally worse, but they do not confirm that this is what actually causes the model to fail on those nodes.

The only model-behaviour observation so far is the influence analysis for a single node (node 1362, degree 22). That is one node, one run, one split. It is a compelling data point but not yet a trend.

#### What is missing to close the causal argument

To move from "structural metrics suggest anomalous high-degree nodes receive worse signal" to "this is why the model fails on them specifically", three things are needed:

1. **Influence analysis across many anomalous high-degree nodes.** A single node could be an outlier. Running the analysis across all misclassified mid/high-degree nodes (across multiple selected degrees and runs) and showing that same_class_influence ≈ 0 is a consistent pattern — not a coincidence — would establish it as a systematic model behaviour rather than a one-off observation. Crucially, this should be contrasted against correctly classified high-degree nodes of similar degree to show the influence distribution differs.

2. **Correlation between influence and correctness.** A scatter plot or binned comparison of same_class_influence vs classification outcome (correct vs misclassified) across many test nodes would directly test whether near-zero same-class influence predicts misclassification. If correctly classified high-degree nodes consistently have higher same_class_influence than misclassified ones, the causal link from aggregation bias to the anomalous accuracy drop is established.

3. **Cross-graph validation.** On a heterophilic graph (e.g. Chameleon, Squirrel), more connections do not imply noisier class signal — in fact, the opposite may hold. If the anomalous high-degree misclassifications disappear on heterophilic graphs, that isolates class-mixing (not degree per se) as the true driver. This would be strong evidence that it is specifically neighbourhood heterogeneity that causes the problem, not high degree in isolation.

Until these are done, the current state is: the structural metrics motivate the hypothesis well, and the single influence observation is consistent with it — but the causal chain from neighbourhood structure to model failure for specific anomalous nodes has not yet been demonstrated as a general trend across the data.

---

## 8. Open Questions / To Explore

*(See `NOTES.md` for detailed Q&A on completed analyses. See `TODO.md` for full descriptions of pending investigations.)*

**Structural / aggregation:**
- How do diff-class nodes dominate aggregation numerically? (ratio analysis, per-layer message decomposition, weight-matrix analysis)
- Do diff-class training neighbours have higher degree than same-class ones? *(Partially addressed by §4.8 / §5.10; systematic analysis pending — see `TODO.md` §1)*
- **Cardinality analysis:** does `same_count > diff_count` reliably predict correct classification, and is this consistent across seeds and degree groups? *(See `TODO.md` §1)*

**Feature-space:**
- **Mean feature distance:** does the mean L2/cosine distance between a test node's features and its neighbors' features correlate with accuracy? *(See `TODO.md` §2)*

**Model / optimisation:**
- **Learned weight suppression:** do the trained weight matrices systematically project same-class neighbour features into low-magnitude directions?
- **Hidden dimension sweep:** does GCN hidden dim interact with degree bias, and does larger capacity recover more same-class influence signal? *(See `TODO.md` §3)*

**Broader validation:**
- **Broader study:** replicate findings across multiple random splits and public splits; across datasets (CiteSeer, PubMed, ogbn-arxiv, Chameleon, Squirrel); across model architectures (GCN, GCNII, GAT).
- **Higher hops:** run influence analysis at k=3 and k=4 for the same selected nodes; compare GCN vs GCNII to separate over-smoothing from degree bias.
