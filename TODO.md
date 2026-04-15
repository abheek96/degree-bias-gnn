# Research TODOs

Open investigation tasks that have not yet been implemented. Each item includes the motivating question and a sketch of how to approach it.

---

## 1. Cardinality of training nodes in neighborhood (same vs diff class count)

**Question:** Does a higher number of same-class training nodes compared to diff-class training nodes in a test node's k-hop neighborhood reliably predict correct classification? Is there a count threshold (e.g., same_count > diff_count) that acts as a strong predictor?

**Motivation:** The influence analysis for node 1362 (same_train=4, diff_train=11) suggests the count imbalance matters. But this is one node. A systematic analysis across all test nodes would establish whether cardinality alone — independent of degree or feature content — is a consistent predictor of correct classification.

**Approach:**
- For each test node, compute `same_count` and `diff_count` in the k-hop neighborhood (already computed in `get_training_neighbor_degree_stats`).
- Compute the "class advantage ratio" = `same_count / (same_count + diff_count)`.
- Plot accuracy vs class advantage ratio: do nodes with ratio > 0.5 (same-class majority) classify correctly at a much higher rate?
- Group by test node degree: does the cardinality effect hold across all degree groups, or is it driven by specific degree ranges?
- Check consistency across seeds: is a node that has same_count > diff_count always correctly classified, or does it vary with model initialisation?

---

## 2. Mean average distance as a noise metric for node feature neighborhoods

**Question:** Does the mean average distance between a test node's features and those of its k-hop neighbors correlate with classification accuracy? High feature-space distance from neighbors signals that the node is an outlier in its local region — aggregating dissimilar features may introduce noise rather than signal.

**Motivation:** The current metrics (purity, SPL, degree) describe the *structural* and *label* properties of the neighborhood. But GNNs aggregate *features*, not labels directly. A test node that is structurally central (low SPL, good purity) but whose features are far from its neighbors' features may still receive misleading aggregated signal because the raw feature content is mismatched. This metric bridges the structural and feature-space analyses.

**Approach:**
- For each test node, compute the mean L2 (or cosine) distance between the node's input features `x_v` and those of each k-hop neighbor `x_u`.
- Report this as `mean_feat_dist` per test node.
- Also compute separately for same-class vs diff-class neighbors: `mean_feat_dist_same` and `mean_feat_dist_diff`.
- Plot `mean_feat_dist` vs test node degree (grouped boxplots): does feature distance increase with degree (hub nodes are more feature-diverse neighbors)?
- Plot against classification accuracy: do misclassified nodes have higher `mean_feat_dist` from their same-class neighbors and/or lower from their diff-class neighbors?

---

## 3. Role of hidden dimension in signal propagation

**Question:** Does the choice of hidden dimension in GCN layers affect how well same-class training signal is propagated to high-degree test nodes? A larger hidden dimension provides more capacity to represent class-discriminative directions, but may also dilute the signal if most dimensions are unused.

**Motivation:** The learned weight suppression hypothesis (§7.2) posits that the weight matrices `W^(l)` project same-class neighbor features into low-magnitude directions. The capacity of these projections is bounded by the hidden dimension. With a very small hidden dimension, the model may lack the representational capacity to separate the same-class signal from the diff-class noise. With a larger dimension, it has more capacity but may still fail if the optimization converges to a suppressed solution. This analysis directly tests whether the architectural choice of hidden dimension interacts with the degree-bias phenomenon.

**Approach:**
- Run the existing `acc_vs_degree_by_layers`-style sweep but over hidden dimensions instead of (or in addition to) layer counts.
- Hidden dim values to test: e.g., [16, 32, 64, 128, 256, 512] for GCN on Cora.
- For each hidden dim, record accuracy vs degree.
- Additionally, for each hidden dim, run the influence analysis on the same selected anomalous nodes: does `same_class_influence` increase with hidden dim, suggesting the model recovers more same-class signal with greater capacity?
- Implement via a new config option (e.g., `hidden_dim_sweep: [16, 64, 256, 512]`) and a new plot `plot_acc_vs_degree_by_hidden_dim`.

---

## 4. Literature citations for each structural metric and factor

**Question:** For each metric and structural factor used in this investigation, what is the earliest / most authoritative source that introduces or motivates it in the context of GNNs or graph-based learning?

**Motivation:** The current codebase and documentation describe each metric clearly but without references. Adding citations would (a) situate each choice in the existing literature, (b) make it easier to position this work relative to prior art, and (c) identify which metrics are genuinely novel contributions of this repository vs. established tools being applied in a new context.

**Metrics / factors to source:**

- **Degree-bias in GNNs** — which papers first formally studied accuracy as a function of node degree?
- **Neighbourhood purity** — is this metric used elsewhere under the same or a different name?
- **Average SPL to training nodes** — has this been used as a diagnostic metric in semi-supervised GNN papers?
- **Labelling ratio (fraction with direct labelled neighbour)** — is this formalised in any label-propagation or GNN literature?
- **GCN degree-normalised aggregation** (`1/sqrt(deg_u * deg_v)`) — cite Kipf & Welling (2017) and trace whether the dilution effect on high-degree nodes is noted there or in subsequent work.
- **Influence / sensitivity analysis via Jacobian** — cite the definition (Definition 3.1 in the codebase); identify the paper this definition comes from.
- **Over-smoothing and depth** — cite the relevant over-smoothing literature (Li et al., 2018; Oono & Suzuki, 2020; or others) for the claim that more layers hurt high-degree nodes more.
- **GCNII (initial residual + identity mapping)** — cite Chen et al. (2020).
- **Training-neighbor degree distribution** — is there prior work decomposing the aggregation imbalance into count vs. degree components?

**Approach:**
- For each bullet above, find the primary source and at least one follow-up or survey that uses the same concept.
- Annotate `RESEARCH.md` §3 (Models), §4 (Metrics), and §7 (Narrative) with inline citations.
- Optionally add a `REFERENCES.md` or a references section at the bottom of `RESEARCH.md` with full BibTeX-style entries.

---

## 5. Information-to-noise ratio across homophilous and heterophilous graphs

**Question:** Can we define a per-node and per-graph information-to-noise ratio (INR) that quantifies how much useful (same-class) signal a GNN receives relative to the misleading (diff-class) signal during aggregation? Does this ratio predict classification accuracy better than any single structural metric, and does it generalise across graphs with fundamentally different homophily levels?

**Motivation:** The current metrics (purity, SPL, labelling ratio, cardinality) each capture a different facet of the signal-quality problem, but none combines them into a single interpretable quantity. An INR would:
- Unify the structural narrative: a node with low INR receives more noise than signal regardless of which specific structural factor dominates.
- Be meaningful on heterophilous graphs (e.g. Chameleon, Squirrel) where high purity is not expected — the INR framing adapts naturally because "information" becomes relative to the graph's own homophily baseline rather than an absolute same-class majority.
- Expose degree-bias as a special case: high-degree nodes in homophilous graphs often have lower INR despite having more connections, because the absolute diff-class count grows faster than same-class count.

**Approach — per-node INR:**
- Define same-class signal strength: weighted count of same-class training nodes in the k-hop receptive field, where weights follow GCN normalisation (`1/sqrt(deg_u * deg_v)` per edge, compounded over hops).
- Define diff-class noise strength: same quantity for diff-class training nodes.
- INR(v) = same_signal(v) / (diff_signal(v) + ε).
- Plot INR vs degree, INR vs accuracy; check whether INR > 1 (same-class majority in weighted sense) reliably predicts correct classification.

**Approach — graph-level INR and homophily sweep:**
- Compute mean and distribution of per-node INR for each graph.
- Datasets to include: Cora, CiteSeer, PubMed (homophilous); Chameleon, Squirrel, Actor (heterophilous); ogbn-arxiv (large-scale homophilous).
- For heterophilous graphs, verify that the INR framing still identifies high-accuracy nodes: on heterophilous graphs, accuracy may correlate with low INR (the model learns to exploit diff-class signal) or with a different threshold than 1.0.
- Plot graph-level homophily (edge homophily ratio) vs mean INR vs mean test accuracy: does INR mediate the relationship between homophily and GNN performance?

**Connections to existing metrics:**
- INR ≈ purity when all training nodes are equidistant (1 hop) and have equal degree.
- INR diverges from purity when degree normalisation is significant (high-degree training nodes contribute less per edge).
- INR can be seen as a continuous generalisation of the binary same_count > diff_count criterion from TODO §1.

---

## 6. Validation node analysis: degree-wise performance and structural properties

**Question:** Do validation nodes exhibit the same degree-bias patterns as test nodes? Are the global and local structural properties of val nodes systematically different from those of test nodes, and do those differences explain any gap in degree-wise accuracy between the two splits?

**Motivation:** All current analyses (accuracy vs degree, purity, SPL, labelling ratio, influence disparity, feature similarity) are run exclusively on test nodes. Since the best checkpoint is selected based on val accuracy, understanding how val nodes differ structurally from test nodes is important — if val nodes are structurally easier (e.g. lower average degree, higher purity, shorter SPL to training nodes), the checkpoint selection criterion may be systematically biased towards configurations that generalise well to easy nodes but not hard ones.

**Approach:**
- Replicate all existing degree-wise analyses for val nodes: accuracy vs degree, neighbourhood purity, SPL to training nodes, labelling ratio, cardinality, influence disparity, feature similarity delta.
- Compute global structural properties for val vs test nodes side by side: degree distribution, mean/median SPL to any training node, mean purity at k=1 and k=2, labelling ratio, class balance.
- Plot val vs test comparison for each metric: overlaid lines or grouped bars, one panel per metric.
- Check whether degree ranges well-represented in val are under-represented in test and vice versa — if so, the checkpoint may be optimised for a different degree regime than the one being evaluated.

**Best-checkpoint analysis:**
- The best model checkpoint is selected by val accuracy, so the structural properties of val nodes directly shape which checkpoint is saved. Analyse the val nodes specifically at the best checkpoint epoch: do correctly-classified val nodes at that epoch have systematically lower degree, higher purity, or shorter SPL than the val nodes that are misclassified?
- Check whether val nodes are randomly sampled (random split) or fixed (public split) and whether their structural profile changes across seeds — if val nodes happen to be structurally easy (e.g. low degree, high purity), the checkpoint selection is biased toward configurations that work well for easy nodes.
- For random splits, note that val nodes are drawn uniformly from the non-training remainder with no stratification by degree or class, so the val set can vary substantially in structural difficulty across seeds.
- For public splits with CC enabled, val nodes outside the largest connected component are silently dropped with no rebalancing, potentially skewing the val set toward denser, more central nodes.

## 7. Hyperparameter grid search

**Question:** Do the degree-bias observations hold robustly across hyperparameter configurations, or are they artefacts of a specific setting?

**Motivation:** All current analyses use a fixed hyperparameter set. If the degree-bias pattern disappears or strengthens under different configurations (lr, hidden dim, dropout, num_layers), the observation is not robust. A grid search establishes which hyperparameter regimes amplify or suppress the bias, and ensures the best-performing configuration is used as the reference point for all other analyses.

**Approach:**
- Grid over: `lr ∈ [1e-3, 5e-3, 1e-2]`, `hidden_dim ∈ [64, 256, 512]`, `dropout ∈ [0.0, 0.3, 0.5, 0.7]`, `num_layers ∈ [2, 3, 4]`.
- Select best configuration by mean val accuracy across seeds.
- Re-run all degree-wise analyses on the best configuration and compare degree-bias profiles against the default configuration.

---

## 8. Confirm observations with pre-trained models from related works

**Question:** Do the degree-bias observations generalise beyond this codebase, or are they an artefact of this specific training setup?

**Motivation:** Using a pre-trained, best-performing model from related work (e.g. MPNN, GCNII, or a model from the LINKX/GLCN family) as a fixed feature extractor removes training variability. If the same degree-bias pattern appears in the predictions of an externally trained model, the observation is a property of the graph structure and aggregation mechanism — not of this training loop.

**Key observation (current finding):** Analyses on the public split do not yield clear or interpretable conclusions — the degree-bias signal is weak or absent. The random split, by contrast, produces coherent and relevant results where degree-bias patterns emerge clearly. This makes the random split the primary setting for all analyses going forward. The public split's lack of signal is likely due to its fixed, non-representative assignment of training/val/test nodes (only 20 training nodes per class, test nodes concentrated in specific degree ranges), which confounds the structural signal. The random split distributes nodes more uniformly across degree ranges, making degree-bias effects visible.

**Secondary observation (public split + CC, tuned MPNN weights, 10 runs):** Even on the public split, `accuracy_vs_degree_across_runs` shows non-trivial variance across seeds for mid/high-degree nodes (e.g. deg=16). Mid-degree groups do not consistently outperform lower-degree ones — the expected monotonic accuracy-vs-degree trend does not hold reliably. This further supports that the public split is not a clean setting for observing degree-bias: the fixed test node assignment likely places nodes of varying structural quality into the same degree bucket, masking the signal with between-node heterogeneity rather than between-degree differences.

**Approach:**
- Load pre-trained weights from a related-work model (e.g. the saved MPNN weights).
- Run all analyses (accuracy vs degree, purity, SPL, influence disparity, feature similarity) on its predictions without any fine-tuning.
- Focus on random splits as the primary evaluation setting; run public split as a secondary check to document the contrast.
- Investigate why the public split suppresses the degree-bias signal: is it the training node count (20 per class), the specific nodes chosen as test nodes, or the degree distribution of test nodes in the public split?

---

## 9. Additional graph-level and node-level characteristics

**Question:** Are there other structural properties of the graph or individual nodes — beyond degree, purity, SPL, and labelling ratio — that explain why certain nodes are consistently misclassified?

**Motivation:** The current metrics focus on immediate neighborhood quality. But node performance may also be explained by properties such as: position within the graph (core vs periphery), local clustering, betweenness centrality, or ego-graph homophily. Identifying additional predictive properties would deepen the structural narrative and may suggest new mitigation strategies.

**Candidates to explore:**
- **Clustering coefficient** — do nodes in tightly connected local clusters classify better?
- **Betweenness / closeness centrality** — are high-centrality nodes easier or harder to classify?
- **Ego-graph homophily** — homophily ratio within the node's k-hop subgraph vs the global graph homophily.
- **Feature-label alignment** — cosine similarity between a node's features and the mean features of its true class in the training set.
- **Position relative to class boundary** — does the node sit at the boundary between two class regions in feature space or in graph topology?

---

## 10. Confirm the necessity of restricting analysis to the largest connected component

**Question:** Do smaller disconnected components confound the degree-bias signal, and is restricting to the largest connected component (LCC) the right way to isolate the phenomenon?

**Motivation:** Real-world citation/co-authorship graphs often contain a large main component and many small satellite components. Nodes in smaller components are structurally different from LCC nodes in ways that are unrelated to degree-bias per se:
- They tend to have low degree by construction (small components cannot contain high-degree hubs).
- They may lack any training node entirely, so the GNN has no labeled signal to propagate to them — misclassification there is a label-propagation failure, not a degree-bias failure.
- Mixing LCC nodes and satellite-component nodes in the same analysis conflates two distinct failure modes: (a) degree-bias within a well-connected graph, and (b) isolation / no-training-signal effects in disconnected fragments.

The goal of this project is to establish whether degree-bias exists as a structural phenomenon inside a connected graph. Including satellite components muddies this picture because their low degree is an artifact of component size, not of the aggregation mechanism acting on a dense neighborhood.

**Approach:**
- For a dataset without CC filtering applied, compute the component-size distribution: how many components exist, and what fraction of nodes are outside the LCC?
- Profile nodes in smaller components: degree distribution, fraction with at least one training neighbor, classification accuracy. Confirm that these nodes are systematically lower-degree and more often lack training neighbors.
- Run the core degree-bias analysis (accuracy vs degree) twice — once on all nodes, once on LCC-only nodes — and compare the curves. If the low-degree, low-accuracy region is driven primarily by satellite-component nodes, the degree-bias signal in the full-graph analysis is partially artifactual.
- Document the LCC filtering decision in `RESEARCH.md` with this evidence: filtering is not just a convenience but a methodological choice to isolate the aggregation-based degree-bias mechanism from the unrelated no-training-signal failure mode.

## 11. High-degree same-class training node connectivity vs misclassification rate

**Question:** Do degree groups that are predominantly connected to high-degree same-class training nodes show higher misclassification rates? A high-degree same-class training node delivers weaker per-edge signal (lower `1/sqrt(deg_u × deg_v)` weight), so even a correctly-placed training anchor may be insufficient if it has too many neighbors of its own.

**Motivation:** The "curse of high-degree" observation (node 1894) shows that high degree hurts a test node by diluting all incoming edge weights and accumulating diff-class noise from non-training neighbors. A complementary effect operates on the training node side: if the same-class training node in a test node's receptive field is itself high-degree, the signal it delivers per edge is attenuated regardless of the test node's degree. Plotting connectivity to high-degree same-class training nodes against misclassification rate per degree group would establish whether this is a systematic effect across the graph or specific to individual nodes.

**Approach:**
- Define "high-degree training node" as a training node with degree above some threshold (e.g. median training node degree, or a fixed value like deg ≥ 10).
- For each test node, check whether its k-hop receptive field contains at least one same-class training node, and if so, whether that training node is high-degree.
- Group test nodes by their own degree and compute: (a) fraction connected to a high-degree same-class training node, (b) misclassification rate.
- Plot both on the same axes per degree group — bars or lines for the connectivity fraction, overlaid line for misclassification rate.
- Check whether degree groups with higher connectivity to high-degree same-class training nodes also show higher misclassification rates, suggesting the training node's own degree is a confounding factor beyond just presence/absence of a same-class anchor.

---

## 12. Run-consistent node selection for influence analysis

**Question:** The influence analysis currently uses the model and predictions from the last training run only. Is the misclassification of a high-degree node a consistent property of the trained model, or does it vary across seeds?

**Motivation:** With 5 runs and different random initialisations, a node may be misclassified in some runs but not others. Running influence analysis on a single (last) run risks analysing a node that is misclassified by chance in that seed but not in general — or missing a node that is consistently misclassified across all seeds and therefore a more reliable anomaly. The target node for influence analysis should be selected based on cross-run consistency, not hard-coded.

**Approach:**
- After all training runs complete, identify test nodes that are misclassified in every run (or in a majority of runs, e.g. ≥ 4 of 5).
- Among those consistently misclassified nodes, filter to high-degree nodes (e.g. degree ≥ some threshold, or the top-k by degree).
- Run influence analysis on the selected nodes using the last run's model (as now), but log which run was used and how many runs agreed on the misclassification.
- Optionally: run influence analysis on each run's model independently for the consistently misclassified nodes, and aggregate the per-run influence scores (mean ± std across runs) to identify robust training-node influence patterns.
- The node selection logic should replace the hard-coded `influence_nodes` / `influence_degrees` config fields, or at minimum augment them with an auto-selection mode.

*Last updated: 2026-04-15*
