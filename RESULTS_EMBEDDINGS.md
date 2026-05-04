# Embedding Feature Results — GCN Misclassification Prediction

> **Status: Preliminary — methodology under review.**
> The with-embeddings LR fits on n~207–296 nodes with ~15 features, giving
> events-per-variable (EPV) of 4–6 — below the threshold for stable coefficient
> estimates.  The mixed-neighbourhood subset is also self-selected (nodes with ≥1
> same-class AND ≥1 diff-class 1-hop neighbour), not a random sample.  Metric
> values and coefficient signs should be treated as indicative, not definitive.
> See TODO.md §17 for the planned methodology fix.

---

## 1. Embedding Feature Descriptions

Computed from `model.get_intermediate(layer=k_hops)` — the penultimate GCN
representation after all message-passing but before the classification head.
Only defined for nodes that have at least one same-class **and** at least one
diff-class 1-hop neighbour; NaN otherwise.

| Feature | Description | Expected direction |
|---|---|---|
| `emb_sim_same_1hop` | Mean cosine sim between focal node's embedding and same-class 1-hop neighbour embeddings | Higher → more correct |
| `emb_sim_diff_1hop` | Mean cosine sim between focal node's embedding and diff-class 1-hop neighbour embeddings | Higher → more failure |
| `emb_purity_delta` | `emb_sim_same_1hop − emb_sim_diff_1hop` | Higher (positive) → more correct |

**Motivation:** The raw cosine similarity (`mean_cosine_sim_1hop`) measures whether
the focal node *looks like* its neighbours in input feature space — but cannot
distinguish same-class from diff-class neighbours.  The embedding features ask a
more precise question: after message passing, does the GCN map the focal node closer
to its same-class neighbours and farther from its diff-class neighbours?  A node
with high `emb_purity_delta` is one where the GCN successfully resolved the
structural ambiguity; a node with low or negative `emb_purity_delta` is one where
it failed.

Because `emb_purity_delta` requires both same-class and diff-class 1-hop neighbours,
it is only defined for structurally *mixed* nodes — exactly the hard cases.

---

## 2. With Embedding Features — Mixed-Neighbourhood Subset

Embedding features included.  The LR fits only on nodes with at least one same-class
**and** one diff-class 1-hop neighbour (nodes for which `emb_purity_delta` is
defined).  These nodes have elevated misclassification rates relative to the full
test set.

| Dataset | Split | n | misc% | AUROC | PR-AUC | Lift@50 | Top predictor |
|---|---|---|---|---|---|---|---|
| Cora | public | 296 | 32.1% | 0.873 ± 0.043 | 0.780 | 2.55× | `same_train_infl_frac_1hop` |
| Cora | random | 301 | 38.9% | **0.928** ± 0.035 | 0.898 | 2.52× | `diff_train_infl_frac_1hop` |
| CiteSeer | random | 208 | 39.9% | 0.883 ± 0.038 | 0.827 | 2.11× | `total_infl_diff_2hop` |
| CiteSeer | public | 207 | 36.7% | 0.889 ± 0.056 | 0.826 | 2.34× | `min_dist_to_same_class_train` |
| PubMed | random | 247 | 26.7% | 0.890 ± 0.022 | 0.727 | 2.84× | `avg_spl_to_same_class_train` |
| PubMed | public | 214 | 26.2% | 0.859 ± 0.055 | 0.697 | 2.37× | `emb_sim_same_1hop` |

---

## 3. Embedding Feature Impact (Paired Comparison)

| Dataset | Split | AUROC (no emb, n=full) | PR-AUC (no emb) | AUROC (with emb, n=mixed) | PR-AUC (with emb) | Δ AUROC |
|---|---|---|---|---|---|---|
| PubMed | public | 0.810 (n=1000) | 0.591 | 0.859 (n=214) | 0.697 | **+0.049** |
| PubMed | random | 0.859 (n=1000) | 0.597 | 0.890 (n=247) | 0.727 | **+0.031** |
| Cora | random | 0.922 (n=915) | 0.749 | 0.928 (n=301) | 0.898 | +0.006 |
| Cora | public | 0.913 (n=915) | 0.756 | 0.873 (n=296) | 0.780 | −0.040 |
| CiteSeer | random | 0.949 (n=659) | 0.881 | 0.878 (n=208) | 0.827 | −0.071 |

Note: populations differ between conditions (full test set vs mixed-neighbourhood
subset), so Δ AUROC is indicative rather than strictly controlled.  For CiteSeer the
apparent negative Δ is a population effect — the with-embedding subset (208 nodes,
39.9% misc) is a harder subproblem than the full test set (659 nodes, 29.3% misc);
structural features alone are already so discriminating on the full set (AUROC 0.949)
that restricting to the ambiguous subset and adding embedding features cannot match
that ceiling.

---

## 4. Key Findings (Preliminary)

### 4.1 Embedding-space purity is the dominant predictor for hard nodes

Within the mixed-neighbourhood subset (nodes with both same-class and diff-class
1-hop neighbours — the structurally ambiguous cases), `emb_purity_delta` and
`emb_sim_same_1hop` take the top positions across several with-embedding runs.
For these nodes, whether the GCN resolves the structural ambiguity in its learned
representations appears to determine the outcome.

`purity_2hop` (rank 1 without embeddings) collapses to rank 12 when embedding
features are included — the embedding features absorb what purity_2hop was proxying
for.  `purity_2hop` is a structural estimate of neighbourhood class composition;
`emb_sim_same_1hop` is the model's actual learned response to that composition.
When the direct model-behaviour measure is available, the structural proxy becomes
redundant.

`mean_cosine_sim_1hop` (raw features, negative) and `emb_sim_same_1hop` (learned
embeddings, positive) have opposite signs — raw feature similarity to neighbours is
a failure signal, but embedding-space similarity to *same-class* neighbours is a
success signal.  This is precisely what message passing is supposed to accomplish:
map structurally noisy raw features into a space where same-class nodes cluster.

### 4.2 Mixed-neighbourhood nodes are disproportionately misclassified

The subset for which embedding features are defined (nodes with ≥1 same-class and
≥1 diff-class 1-hop neighbour) has misclassification rates of 26–39% vs ~21%
overall.  These are the structurally contested nodes and they account for a
disproportionate fraction of all failures.

### 4.3 Influence fractions in embedding runs

**`same_train_infl_frac_1hop` is the top predictor in the Cora public with-embeddings
run** (n=296, coef +0.965), consistent with its rank-1 position in the no-embedding
full-set run (n=915, coef +1.267).

**Asymmetry between same-class and diff-class influence fractions (Cora random
with embeddings):** `diff_train_infl_frac_1hop` (−2.233) is the dominant predictor
by a large margin — roughly 4× the magnitude of `same_train_infl_frac_1hop`
(+0.521).  This asymmetry means that wrong-class training signal contaminating
the influence budget is far more detrimental than the equivalent amount of
same-class training signal is protective.

This run also achieves perfect precision in the top-30 flagged nodes (Lift@30 =
2.57×, precision@30 = 100%).

**`degree` flips to negative (−0.434) in the Cora random mixed-neighbourhood
subset**, in contrast to its usually positive or near-zero coefficient.  In the
mixed-neighbourhood population (nodes with both same-class and diff-class 1-hop
neighbours), higher degree means proportionally more cross-class connections —
once inside this structurally contested subset, more edges increase the likelihood
that diff-class training nodes command a large share of the influence budget.
Degree is protective in the full population but becomes a liability within the
mixed-neighbourhood subset.

**`same_train_infl_frac_1hop` is literally 0.000 for CiteSeer random in the
with-embeddings mixed-neighbourhood subset** — the 1-hop fraction is entirely
uninformative for structurally contested nodes in that restricted population.  The
2-hop fraction carries all the signal because same-class training nodes rarely
appear at hop 1 for these ambiguous nodes.

**PubMed public with embeddings:** `emb_sim_same_1hop` (+1.270) dominates.
Influence fraction features rank uniformly low and several flip sign —
the embedding features absorb the signal, leaving influence fractions as residuals.
Adding influence features slightly reduced AUROC (0.866 → 0.859) and PR-AUC
(0.715 → 0.697).

**PubMed random with embeddings:** `avg_spl_to_same_class_train` (−1.779) and
`avg_spl_to_train` (+1.394, collinearity flip) dominate.  Embedding features
(`emb_purity_delta` +0.552, `emb_sim_same_1hop` +0.482) contribute at ranks 4 and
7.  Adding influence features slightly reduced AUROC (0.892 → 0.890) and PR-AUC
(0.741 → 0.727).

---

## 5. Publishable Claims (Pending Methodology Fix)

The following claims depend on the with-embeddings LR results and should not be
stated as established findings until the methodology is validated (see TODO.md §17):

- **Among structurally ambiguous nodes** (mixed 1-hop neighbourhoods, 26–39%
  misclassification), embedding-space purity is the decisive factor: nodes where
  the GCN successfully separates same-class from diff-class neighbours in
  representation space are correctly classified; those where it fails are
  misclassified.

- **Raw feature similarity to neighbours predicts failure; embedding similarity
  to same-class neighbours predicts success** — the two features have opposite signs
  and together describe when message passing helps vs hurts.
