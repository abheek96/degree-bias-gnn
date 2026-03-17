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

*Last updated: 2026-03-17*
