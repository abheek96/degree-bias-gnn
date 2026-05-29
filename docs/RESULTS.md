# Quantitative Results — Predicting GCN Misclassification from Node Features

> Last updated: 2026-05-04

---

## 1. Research Question and Motivation

Qualitative per-node analysis (Jacobian-L1 influence tables) establishes that GCN misclassification
is caused by a combination of factors whose relative importance varies node by node — there is no
single structural explanation that generalises cleanly.  To make this publishable, the analysis is
recast as a prediction problem:

> **Given a set of node-level structural and model-derived features, can we predict which test nodes
> the GCN will fail on?**

A logistic regression (simple, linear, interpretable) is deliberately chosen.  The signed
coefficients (sorted by magnitude) identify which factors drive failure and in which direction,
directly answering the "which features matter?" question for the paper.

**PR-AUC** (average precision, positive = misclassified) is the primary metric for establishing
that misclassification is *systematic*: it measures how precisely failures concentrate at the top
of the predicted-failure ranking.  A high PR-AUC means failures are not scattered randomly — they
are structurally predictable and cluster among the nodes the model flags first.  Because PR-AUC is
sensitive to the minority class and to the baseline rate, it is the honest metric for the
systematicity claim within a dataset.

**AUROC** measures overall discrimination across both classes and is prevalence-independent, making
it suitable for cross-dataset comparisons where baseline misclassification rates differ (20–30%
across Cora, CiteSeer, PubMed).  **Lift@k** is the concrete, interpretable version: "flagging the
top-k nodes finds X× more failures than random."

All results use one checkpoint per condition (run 1, seed 42) and 5-fold stratified cross-validation.

---

## 2. Feature Table

Each row is one test node.  Features are grouped below by type.

### 2.1 Graph-structural (no model needed)

| Feature | Description | Expected direction |
|---|---|---|
| `degree` | In-degree of the test node | Higher → generally more correct |
| `min_dist_to_train` | Minimum SPL to any training node | Ambiguous (see §5) |
| `min_dist_to_same_class_train` | Minimum SPL to a same-class training node | Higher → more failure |
| `avg_spl_to_train` | Average SPL to all training nodes | Ambiguous (see §5) |
| `avg_spl_to_same_class_train` | Average SPL to same-class training nodes | Higher → more failure |
| `purity_1hop` | Fraction of cumulative 1-hop neighbourhood sharing focal class | Higher → more correct |
| `purity_2hop` | Fraction of cumulative 2-hop neighbourhood sharing focal class | Higher → more correct |
| `n_same_train_1hop` | # same-class training nodes in the 1-hop ring | Higher → more correct |
| `n_diff_train_1hop` | # diff-class training nodes in the 1-hop ring | Higher → more failure |
| `n_same_train_2hop` | # same-class training nodes in the 2-hop ring | Higher → more correct |
| `n_diff_train_2hop` | # diff-class training nodes in the 2-hop ring | Higher → more failure |
| `same_train_ratio_1hop` | n_same_train_1hop / \|1-hop ring\| | Higher → more correct |
| `diff_train_ratio_1hop` | n_diff_train_1hop / \|1-hop ring\| | Higher → more failure |
| `same_train_ratio_2hop` | n_same_train_2hop / \|2-hop ring\| | Higher → more correct |
| `diff_train_ratio_2hop` | n_diff_train_2hop / \|2-hop ring\| | Higher → more failure |
| `mean_cosine_sim_1hop` | Mean cosine sim between focal node and 1-hop neighbours (raw features) | Higher → more failure (see §5) |

### 2.2 Jacobian-L1 influence (model-derived, one forward+backward pass per node)

| Feature | Description | Expected direction |
|---|---|---|
| `total_infl_same_1hop` | Sum of Jacobian-L1 influence from all same-class 1-hop nodes | Higher → more correct |
| `total_infl_diff_1hop` | Sum of Jacobian-L1 influence from all diff-class 1-hop nodes | Higher → more failure |
| `total_infl_same_2hop` | Sum of Jacobian-L1 influence from all same-class 2-hop nodes | Higher → more correct |
| `total_infl_diff_2hop` | Sum of Jacobian-L1 influence from all diff-class 2-hop nodes | Higher → more failure |
| `same_train_infl_frac_1hop` | Fraction of total I_x attributable to same-class training nodes at hop 1 | Higher → more correct |
| `diff_train_infl_frac_1hop` | Fraction of total I_x attributable to diff-class training nodes at hop 1 | Higher → more failure |
| `same_train_infl_frac_2hop` | Fraction of total I_x attributable to same-class training nodes at hop 2 | Higher → more correct |
| `diff_train_infl_frac_2hop` | Fraction of total I_x attributable to diff-class training nodes at hop 2 | Higher → more failure |

### 2.3 Embedding-space similarity

See `RESULTS_EMBEDDINGS.md` — methodology under review.

---

## 3. Evaluation Metrics

- **PR-AUC** (average precision, positive = misclassified) — **primary metric for the systematicity
  claim**.  Measures how precisely failures concentrate at the top of the predicted-failure ranking
  across all thresholds.  A PR-AUC well above the baseline misclassification rate (20–30% depending
  on dataset) means failures are not randomly distributed — they are structurally predictable.
  Sensitive to minority-class size, so values are not directly comparable across datasets with
  different baseline rates.
- **Lift@k** — the concrete, interpretable form of PR-AUC: among the top-k nodes ranked by
  predicted failure probability (out-of-fold scores), what fraction are actually misclassified
  relative to the baseline rate?  "Flagging the top-50 nodes finds X× more failures than random."
- **AUROC** — measures overall discrimination of failure vs success; class-symmetric and
  prevalence-independent.  Used for **cross-dataset comparisons** where baseline rates differ.
  Does not directly measure how well failures concentrate among the highest-confidence predictions.

LR uses `class_weight="balanced"` to account for the ~21% minority class.
PR-AUC and Lift@k use out-of-fold `predict_proba` scores (no train-set leakage).

---

## 4. Results

### 4.−3 Feature-subset ablation (Cora · GCN, PR-AUC)

PR-AUC comparison across feature-group ablations on Cora. Baseline (random ranking)
PR-AUC equals the misclassification rate (≈ 0.20–0.21).

> **Scope note.** The table below is a **single** GCN checkpoint; the only resampling is
> the 5-fold stratified CV within that run. To report PR-AUC variance across seeds, use
> `uv run analysis/node_feature_table.py --subset-across-runs`, which computes this same
> subset ablation on every one of the `num_runs` checkpoints and reports each subset's
> PR-AUC as **mean ± std across runs** (`subset_comparison_across_runs.csv` / `.png`).
> _(Across-runs numbers pending a run on the remote server.)_

| Configuration | Group(s) | PR-AUC | Lift over baseline |
|---|---|---|---|
| **Degree (alone)** | single | **0.211** | **≈ 1.0× (at baseline)** |
| Centrality (alone) | single | 0.260 | 1.24× |
| Purity (alone) | single | 0.637 | 3.04× |
| Training proximity (alone) | single | 0.640 | 3.05× |
| Degree + Purity | combo | 0.632 | 3.01× (no gain over Purity) |
| Degree + Training proximity | combo | 0.641 | 3.06× (no gain over Training proximity) |
| All structural (no influence) | combo | 0.715 | 3.41× |
| **Influence (alone)** | single | **0.748** | **3.57×** (highest) |
| Full model | all | 0.739 | 3.52× |

**Observations.**

1. **Degree alone is at baseline.** PR-AUC 0.211 ≈ baseline misc rate; degree as a sole
   feature carries no useful precision lift for identifying misclassified nodes.
2. **Degree is redundant, not merely weak.** Adding degree to Purity changes PR-AUC by
   −0.005 (0.637 → 0.632); adding it to Training proximity changes it by +0.001 (0.640 →
   0.641). Once Purity or Training proximity is in the model, degree contributes no
   additional precision.
3. **Influence alone outperforms every other configuration**, including the full model
   (0.748 vs. 0.739). Jacobian-L1 influence features capture the operative quantity for
   predicting GCN misclassification on Cora more efficiently than structural features do.
4. **All-structural (no influence) achieves 96.8% of the full-model PR-AUC** (0.715 / 0.739).
   When the Jacobian computation is too expensive, structural features alone provide a
   close substitute.
5. **Centrality alone (0.260) is only marginally above degree.** The two pure-topology
   features (degree, centrality) are the weakest single-group predictors by a wide margin.

### 4.−2 Case study: node 1362 (Cora · random split)

Per-node SHAP waterfall decomposition for test node 1362 (Cora, GCN, random split).

| Quantity | Value |
|---|---|
| Degree | 22 |
| `purity_1hop` | 0.77 |
| Model output `f(x)` (log-odds of misclassification) | +1.229 |
| Baseline `E[f(X)]` | −1.40 |
| Actual outcome | **Misclassified** |
| Predicted P(misc) | ≈ 0.77 |

Top SHAP contributors:

| Feature | Value | SHAP | Direction |
|---|---|---|---|
| `degree` | 22 | **−5.18** | Protective (largest single \|SHAP\|) |
| `diff_train_infl_frac_1hop` | 0.031 | **+3.03** | Failure-promoting |
| `eigenvector_centrality` | 0.087 | **+2.72** | Failure-promoting |
| `total_infl_diff_1hop` | 561.24 | +1.48 | Failure-promoting |
| `total_infl_same_1hop` | 1591.10 | −0.80 | Protective |
| `n_same_train_2hop` | 4 | −0.76 | Protective |
| `n_diff_train_1hop` | 2 | −0.71 | Protective |
| `diff_train_ratio_1hop` | 0.091 | −0.55 | Protective |
| `total_infl_diff_2hop` | 726.14 | +0.53 | Failure-promoting |
| 17 other features (aggregate) | — | +1.23 | Net failure-promoting |

**Observations.**

1. **Degree contributes the largest protective signal of any single feature** (−5.18) — the
   model would predict this node to be correct based on degree alone.
2. **That protective signal is overwhelmed by three competing features summing to +7.23**:
   the diff-class influence fraction at 1-hop, eigenvector centrality, and the raw cross-class
   influence sum.
3. **Purity features do not rescue the prediction.** `purity_1hop = 0.77` is high, yet
   purity does not appear among the top contributors — it is absorbed into the "17 other
   features" residual.
4. **The model gets it right.** Final log-odds +1.229 → P(misc) ≈ 0.77; the node is in fact
   misclassified by the GCN.
5. **Single counterexample to a degree-only account.** A node with degree 22 (well above the
   high-degree plateau threshold of 11 under the public split) is a confident,
   correctly-predicted GCN failure under the random split, driven by the class composition of
   its Jacobian influence budget rather than by any deficit of connectivity.

### 4.−1 SHAP feature importance (Cora · GCN · random split, OOF across 5 folds)

Out-of-fold SHAP values for the full-feature LR (no embeddings) on Cora with stratified
random splits; positive SHAP increases P(misclassified). Features ordered by mean |SHAP|.

| Rank | Feature | Direction (high value →) | Notes |
|---|---|---|---|
| 1 | `total_infl_diff_1hop` | ↑ misc | Raw cross-class influence dominates aggregate ranking |
| 2 | `total_infl_diff_2hop` | ↑ misc | Same pattern at 2-hop |
| 3 | `same_train_infl_frac_2hop` | ↓ misc | Protective; long negative tail |
| 4 | **`degree`** | ↓ misc on average | High variance; high-degree nodes can still be on the positive side |
| 5 | `same_train_infl_frac_1hop` | ↓ misc | Long negative tail |
| 6 | `purity_2hop` | ↓ misc | Outranked by 4 influence features |
| 7 | `same_train_ratio_1hop` | ↓ misc | |
| 8 | `n_same_train_2hop` | ↓ misc | |
| 9 | `diff_train_infl_frac_1hop` | ↑ misc | **Largest individual positive SHAP values** (tail to ≈+20) |
| 10 | `diff_train_infl_frac_2hop` | ↑ misc | |
| … | … | … | `purity_1hop` ranks last; signal lives in `purity_2hop` |

**Observations.**

1. **Influence features dominate the top of the ranking.** Four of the top five features (and
   six of the top ten) are Jacobian-L1 influence quantities — either raw totals or fractions
   of the influence budget contributed by same/diff-class training nodes.
2. **Degree is in the top tier but not the top.** It ranks 4th by mean |SHAP|, behind two
   raw-cross-class-influence features and one same-class-influence-fraction feature.
3. **Degree's effect is heterogeneous.** High-degree nodes (red) cluster slightly left of
   zero (protective) but the red distribution spans both sides — a non-trivial subset of
   high-degree nodes receive *positive* SHAP from their degree, meaning their high degree is
   contributing to (not preventing) their misclassification. This is the SHAP-level fingerprint
   of the high-degree failure case studies in §4.2.4.
4. **`diff_train_infl_frac_1hop` is the strongest per-node failure driver.** Its mean |SHAP|
   rank is only 9, but its individual SHAP values reach ≈+20 — the most extreme single-feature
   contribution to predicted misclassification probability for any node in the dataset.
5. **`purity_2hop` ranks 6th, `purity_1hop` ranks last.** The neighbourhood-purity signal is
   carried entirely by the 2-hop variant; the 1-hop fraction adds almost nothing once 2-hop
   purity is in the model.

### 4.0 Accuracy vs. node degree across runs (Cora, public vs. random)

Per-degree test accuracy across 10 seeds, Cora · GCN · CC filter, n = 915 test nodes.
Mean test accuracy is comparable across splits (public 81.3%, random 79.9%), confirming
that the split does not materially change model quality — only the shape of the
degree–accuracy relationship.

| Degree | n nodes | Public median acc | Public IQR | Random median acc | Random IQR |
|---|---|---|---|---|---|
| 1  | ~125 | 0.74 | 0.72–0.77 | 0.78 | 0.77–0.79 |
| 2  | ~205 | 0.78 | 0.77–0.79 | 0.78 | 0.77–0.79 |
| 3  | ~205 | 0.83 | 0.82–0.84 | 0.80 | 0.78–0.81 |
| 4  | ~130 | 0.86 | 0.84–0.87 | 0.86 | 0.83–0.87 |
| 5  | ~95  | 0.88 | 0.85–0.89 | 0.87 | 0.84–0.89 |
| 6  | ~30  | 0.76 | 0.75–0.78 | 0.72 | 0.70–0.75 |
| 7  | ~15  | 0.79 | 0.76–0.81 | 0.77 | 0.74–0.80 |
| 8  | ~12  | 0.79 | 0.74–0.83 | 0.91 | 0.85–0.93 |
| 9–10 | ~10 | 0.71–0.91 (wide) | wide | 0.73 | 0.65–0.82 (wide) |
| 11–12 | ~5 | ≈1.00 (tight) | tight | 0.67–1.00 (bimodal) | full range |
| ≥13 | <5 each | ≈1.00 | tight | mostly 1.00; **deg-16 ≈ 0.33** | persistent failure |

**Observations.**

1. **Low-degree rise (1→5) is present in both splits.** Both rise from ~74–78% at degree 1
   to ~87–88% at degree 5. This is the trend the degree-bias literature reports — and
   it survives the random split.
2. **Non-monotonic dip at degree 6 is also present in both splits** (public 76%, random 72%).
   The dip is therefore a graph property, not a split artefact.
3. **The high-degree plateau is split-dependent.** Under the public split, every node with
   degree ≥ 11 is classified correctly in every seed (IQR collapses to 0). Under the random
   split, this plateau breaks: degree 11–12 and 17 show full-range bimodal outcomes across
   seeds, and a node at degree 16 is misclassified in every seed (~33% accuracy).
4. **Mean accuracy is preserved across splits** (81.3% vs. 79.9%): the random split does not
   produce a worse model, only a less predictable degree–accuracy relationship.

The high-degree plateau under the public split is consistent with the
training-placement confound hypothesis: with training nodes fixed at the Planetoid
positions, high-degree hubs sit closer to the training set than they do under random
placement, so degree becomes a proxy for training-node proximity rather than an
independent predictor.

### 4.1 Without embedding features — full test set

Embedding features excluded (all NaN columns dropped automatically).  The LR fits on all test nodes.

| Dataset | Split | n | misc% | AUROC | PR-AUC | Lift@50 | Top predictor |
|---|---|---|---|---|---|---|---|
| Cora | public | 915 | 20.3% | **0.913** ± 0.014 | 0.756 | 4.53× | `same_train_infl_frac_1hop` |
| Cora | random | 915 | 20.9% | **0.922** ± 0.032 | 0.749 | 4.02× | `diff_train_infl_frac_1hop` |
| CiteSeer | random | 659 | 29.3% | **0.949** ± 0.010 | 0.881 | 3.35× | `min_dist_to_same_class_train` |
| PubMed | random | 1000 | 21.2% | 0.859 ± 0.020 | 0.597 | 3.49× | `purity_2hop` |
| PubMed | public | 1000 | 23.4% | 0.810 ± 0.038 | 0.591 | 3.50× | `degree` |

### 4.2 With embedding features

See `RESULTS_EMBEDDINGS.md` — methodology under review (EPV concern, self-selected subset).

### 4.4 Ablation baselines — Cora public (no embeddings, n=915)

Two minimal-feature baselines establish the contribution of richer features.

**Degree-only LR** (`--features degree`): uses degree as the sole predictor.
**Purity-only LR** (`--features purity_1hop,purity_2hop`): uses the two neighbourhood purity
fractions as the sole predictors.

| Features | AUROC | PR-AUC | Lift@50 |
|---|---|---|---|
| Degree only (1 feature) | 0.566 ± 0.035 | 0.245 | 1.38× |
| Purity only (2 features) | 0.813 ± 0.030 | 0.629 | **4.43×** |
| Full structural + influence (16 features) | **0.913** ± 0.014 | **0.756** | 4.53× |

**Univariate AUROC** (feature used directly as ranking score, no model) confirms degree's weakness
is not a linearity artefact of logistic regression:

| Feature | Univariate AUROC |
|---|---|
| `purity_2hop` | 0.808 |
| `purity_1hop` | 0.800 |
| `min_dist_to_same_class_train` | 0.757 |
| `same_train_ratio_2hop` | 0.705 |
| … | … |
| `degree` | 0.565 |
| `avg_spl_to_train` | 0.512 |

**Interpretation:**

- Degree is near-random as a predictor (univariate AUROC 0.565, LR AUROC 0.566), confirming it
  carries almost no useful information about which individual nodes the GCN will fail on, despite
  its well-known aggregate correlation with accuracy by degree group.
- Two purity features alone recover **98% of the full model's Lift@50** (4.43× vs 4.53×).
  The nodes the model is most confident will be misclassified are identified almost as well by
  neighbourhood class composition alone as by the full 16-feature set.
- The full feature set adds +0.10 AUROC and +0.13 PR-AUC over purity-only, improving the ranking
  of the ambiguous middle — nodes with intermediate purity where SPL, training-node density, and
  influence fraction features resolve uncertainty.
- This yields a clean three-tier structure: **degree → near-random (0.57); purity → identifies
  the obvious cases (0.81, Lift@50 4.43×); full structural + influence → resolves the ambiguous
  middle (0.91).**

### 4.5 Ablation baselines — Cora random vs public split comparison

Running the same three baselines on the random split (n=915, baseline rate 20.9%) and comparing
against the public split results above:

| Features | Public AUROC | Public PR-AUC | Random AUROC | Random PR-AUC |
|---|---|---|---|---|
| Degree only | 0.561 | 0.245 | **0.457** | 0.211 |
| Purity only | 0.810 | 0.629 | 0.841 | 0.622 |
| Full features | 0.913 | 0.756 | **0.920** | 0.749 |

**Degree drops below 0.5 AUROC on the random split (0.457).** The ROC curve dips visibly below
the diagonal — degree is slightly anticorrelated with misclassification, meaning high-degree nodes
are marginally more likely to be correctly classified.  In the public split, fixed training-node
positions create weak degree-correlated structure (AUROC 0.561); in the random split, training
nodes are scattered uniformly, breaking even that residual signal.  This confirms that degree's
public-split predictive value was an artefact of training-node placement, not an intrinsic
relationship between degree and GCN failure.

**Purity is robust to split type.** AUROC improves slightly (0.810 → 0.841) while PR-AUC is
nearly identical (0.629 → 0.622).  Neighbourhood class composition predicts failure regardless of
how training nodes are assigned — purity is a property of the graph's label structure, not of the
split.

**Full features improves AUROC (0.913 → 0.920) but PR-AUC is slightly lower (0.756 → 0.749).**
Higher overall discrimination coexists with marginally weaker minority-class detection at medium
recall.  The random split's influence fraction features (`diff_train_infl_frac_1hop` dominates at
coef −1.918) add strong signal that lifts AUROC, but the slightly higher baseline rate (0.209 vs
0.203) makes precision harder to maintain across the full recall range.

**The gap between purity-only and full features is consistent across both splits.**  In both cases,
the two curves start at near-identical precision (~1.0) at low recall, then full features maintains
substantially higher precision from recall ≈ 0.1 onward.  The richer features resolve the
ambiguous middle; the obvious failures (low-purity nodes) are identified by purity alone.

---

## 5. Key Findings

### 5.1 GCN failure is strongly predictable from structure alone

**PR-AUC 0.59–0.88 from a linear classifier on structural/model-derived features establishes that
misclassification is systematic** — failures concentrate among structurally identifiable nodes, not
scattered randomly.  PR-AUC is the primary evidence for this claim: it measures how precisely
failures cluster at the top of the predicted-failure ranking, directly within each dataset at its
own baseline rate.  The result holds across three datasets (Cora, CiteSeer, PubMed) and multiple
split regimes.

For cross-dataset comparison, AUROC (prevalence-independent) ranges 0.80–0.94.  CiteSeer without
embeddings achieves the highest AUROC of all conditions (0.949) and the highest PR-AUC (0.881),
despite having the highest baseline misclassification rate (29.6%) — lower homophily makes failure
more structurally concentrated, not less.

### 5.2 Predictability varies with graph size and homophily

Cora (AUROC 0.911) is smaller and sparser; the fixed placement of 20 training nodes per class
creates strong and systematic variation in 2-hop training composition across test nodes.  PubMed
is larger (~19K nodes), spreading the same 20-per-class labels more thinly and producing weaker
structural signals (AUROC 0.799–0.861).

CiteSeer occupies a distinct position: its lower graph homophily (~74% vs ~81% for Cora/PubMed)
means the 2-hop ring is more class-heterogeneous (purity_2hop median = 0.789 vs ≈1.0 for Cora),
making the purity signal stronger and more discriminating.  Its higher baseline misclassification
rate (29.6% vs ~21%) reflects this lower homophily, yet structural features alone achieve the
highest AUROC of all conditions (0.949 without embeddings) — counterintuitively, lower homophily
makes failure more structurally predictable, not less.

### 5.3 Predictability is robust to training split within Cora (0.911 on both splits)

The identical AUROC on Cora public and Cora random splits confirms the finding is not an artefact
of which specific nodes are labelled — it reflects a stable structural property of the graph.

### 5.4 purity_2hop is a consistent structural predictor; degree dominates in thin-signal conditions

`purity_2hop` is the top or near-top predictor in the no-embedding runs, ranking 2nd on Cora
public/random and PubMed public, and 1st on PubMed random and CiteSeer random.  Higher fraction of
same-class nodes in the cumulative 2-hop neighbourhood → more likely correct.  It is the most
stable structural signal across datasets.

With the influence fraction features added, `purity_2hop` is displaced to rank 2 by more
informative predictors in several conditions:

- **Cora public** (no emb): `same_train_infl_frac_1hop` (+1.267, rank 1) — an influence-weighted
  variant of the same concept; the Jacobian budget share is more discriminating than the raw composition.
- **Cora random** (no emb): `diff_train_infl_frac_1hop` (−1.918, rank 1) — wrong-class training
  influence dominates.
- **PubMed public** (no emb): `degree` (+1.294, rank 1) — with training signal too thin to make
  fractions discriminating, degree emerges as the dominant proxy for structural advantage.
  `purity_2hop` still ranks 2nd (+0.789), confirming it remains an important predictor even when
  not top-ranked.
- **CiteSeer random** (no emb): `min_dist_to_same_class_train` (−1.362, rank 1) — distance is the
  binding constraint under lower homophily and sparse training signal.

### 5.5 Proximity to same-class training signal is consistently critical

`min_dist_to_same_class_train` is negative across all runs — higher minimum hop-distance to the
nearest same-class training node → more likely misclassified.  The same-class label signal
must travel more hops, getting diluted and mixed with cross-class signal at each intermediate
aggregation step.

The paired SPL features (`avg_spl_to_train` positive, `avg_spl_to_same_class_train` negative) have
large opposite-sign coefficients and are collinear — interpret them jointly: being close to training
nodes of *any* class is insufficient or even counterproductive when those nodes are the wrong class.
What matters is proximity specifically to *same-class* training signal.

### 5.6 Raw feature cosine similarity with neighbours predicts failure

`mean_cosine_sim_1hop` (raw input features) is negative in most runs — a node whose features are
more similar to its 1-hop neighbours in raw feature space is *more likely* to fail.  The
mechanism: if a node's raw features resemble its neighbours, and those neighbours are predominantly
the wrong class (low purity), the model cannot distinguish the focal node from its heterogeneous
neighbourhood, pulling its representation toward the wrong class.  This is consistent with
`mean_cosine_sim_1hop` and `purity` having opposite effects: cosine similarity amplifies the
purity signal (see §6).

### 5.7 Degree rises from weak proxy to dominant predictor in thin-signal conditions

`degree` is weak when structural features are highly informative (rank 14 on Cora random, coef
+0.09).  It becomes moderately
prominent when those features provide less signal (rank 7 on PubMed random, +0.39; rank 3 on Cora
public no-embedding, +0.786).  The extreme case is **PubMed public without embeddings** with the
new influence features: `degree` (+1.294) is the **top predictor of all 24 features**, outranking
`purity_2hop` (+0.789) and `min_dist_to_same_class_train` (−0.628).

The mechanism reflects a hierarchy of signal quality.  When influence fractions are discriminating
(Cora, where training nodes are dense and nearby), they absorb degree's proxy role — degree then
becomes residual.  When fractions are weak (PubMed, 20 training nodes per class across ~19K nodes),
the Jacobian budget competition is too diffuse to discriminate: degree falls back as the dominant
quantity-of-connections signal, since more neighbours statistically means more chance of reaching a
training node.  This is the same quantity-of-connections effect noted in §5.10, but amplified when
no influence-based feature can capture training-signal quality.

**Degree was a proxy for structural advantage, not the cause of it** — but when better proxies are
unavailable (thin training signal, PubMed regime), it is the best available one.  The positive
coefficient in the full-set runs is not spurious; it is a population-level confound that carries
real predictive content whenever the signal-quality features cannot compete.

### 5.8 Degree × purity interaction: high degree amplifies wrong-class signal when homophily is low

It is tempting to assume high-degree nodes always benefit from richer training signal.  This is wrong
in the general case.  GCN's symmetric normalisation scales each edge (i→j) by 1/√(deg_i · deg_j),
and this scaling applies equally to same-class and wrong-class signals — it does not selectively
amplify the correct class.

For a high-degree node with low neighbourhood purity (say 40% same-class), the aggregate
representation is pulled toward a weighted mixture that reflects the class imbalance in the
neighbourhood, with the imbalance *confirmed* by many contributing neighbours rather than cancelled.
This can be worse than a low-degree node in the same purity regime, where only one or two
neighbours contribute and the model still has more relative weight from the focal node's own
features.

Concretely:

| Degree | Purity | Consequence |
|---|---|---|
| High | High | Many same-class signals → consistently correct |
| High | Low | Many wrong-class signals, each scaled but collectively a confident wrong aggregate |
| Low | High | Few signals, right class — often correct |
| Low | Low | One or two wrong-class signals dominate at high per-edge weight |

The positive degree coefficient in the LR (+0.09 to +0.39) is a population-level confound:
high-degree nodes statistically occupy denser, more homophilic graph regions in Cora and PubMed.
Once purity_2hop and same_train_ratio are controlled, degree drops near zero — confirming it is
a proxy for neighbourhood quality, not a direct protective mechanism.  The interaction test (§6)
directly supports this: the (high cosine sim × low purity) quadrant, which captures exactly the
high-degree / heterophilic case, has misclassification rates 2–4× the baseline.

### 5.9 Low-degree nodes do not necessarily have worse structural features — 2-hop expansion can compensate

Low-degree nodes face a qualitatively different structural situation from high-degree nodes, but not
a uniformly worse one.

**1-hop:** The signal is narrow and high-variance.  A degree-1 node has purity_1hop ∈ {0, 1} —
perfectly homophilic or perfectly heterophilic depending on its single neighbour's class.  One
wrong-class direct neighbour produces a single strong, un-averaged wrong signal; one same-class
neighbour produces a strong correct signal.

**2-hop:** The 2-hop neighbourhood can expand dramatically even for low-degree nodes, depending on
the degree of their 1-hop neighbours.  If a degree-1 node's single neighbour is a hub (degree 30),
then the degree-1 node's 2-hop neighbourhood includes ~30 additional nodes — comparable to or
larger than that of many intermediate-degree nodes.  `purity_2hop` and `same_train_ratio_2hop`
for such a node may be no worse than for a degree-10 node in a sparser region.

**Where low-degree nodes genuinely suffer:**
1. Their single 1-hop neighbour is wrong-class — they receive one strong undiluted wrong-class
   signal with no aggregation to average it out.
2. They are situated at the periphery of the graph, far from all training nodes — `min_dist_to_train`
   and `avg_spl_to_train` are high, so even multi-hop propagation cannot deliver training signal.

Both are structural position problems, not degree problems per se.  A low-degree node well-placed
in a homophilic, training-node-adjacent region of the graph can be as easy to classify correctly
as a high-degree node in the same region.  The degree-bias narrative applies most cleanly to the
subset of low-degree nodes that are simultaneously peripheral and in heterophilic local
neighbourhoods — a correlation that holds statistically but is not a universal rule.

### 5.10 CiteSeer: lower homophily makes purity_2hop near-sufficient for predicting correctness

CiteSeer's graph homophily (~74%) is meaningfully lower than Cora/PubMed (~80–81%).  This has two
consequences:

1. **purity_2hop becomes a much more discriminating feature.**  On Cora, more than half of test
   nodes have purity_2hop ≈ 1.0 — most 2-hop rings are nearly pure, so the median split does not
   separate nodes with meaningfully different structural risk.  On CiteSeer, purity_2hop median =
   0.789: the "high purity" half genuinely has a clean 2-hop neighbourhood and the "low purity"
   half is substantially mixed.  The interaction test consequence is stark: nodes with high
   purity_2hop fail at only **1.8%**; those with low purity_2hop fail at **55–57%** — roughly
   30× the rate.  No other single feature-split produces such a clean separation in this analysis.

2. **`min_dist_to_same_class_train` becomes the top LR predictor** (coef −1.24), overtaking
   `purity_2hop` (+0.87, second).  In a lower-homophily graph, reaching the nearest same-class
   training node requires traversing cross-class nodes more often, increasing the number of
   conflicting aggregation steps and making proximity to same-class supervision the critical
   bottleneck.

3. **`n_diff_train_2hop` (third, coef −0.78)** is more prominent in CiteSeer than in Cora/PubMed.
   Lower homophily increases the expected number of wrong-class training nodes in the 2-hop ring,
   so this count carries more signal.

**Without embeddings (full test set, n=659), the structural signal is extraordinarily strong.**
AUROC reaches 0.949 — the highest of all conditions across all datasets — with PR-AUC 0.881 and
Lift@50 = 3.35× (96% precision in the top-50 flagged nodes).  With the influence fraction features
included, the top predictors are: `min_dist_to_same_class_train` (−1.362), `purity_2hop` (+1.041),
`same_train_infl_frac_2hop` (+0.939, rank 4), `diff_train_infl_frac_2hop` (−0.858, rank 5),
`total_infl_diff_2hop` (−0.758, rank 7), and `same_train_infl_frac_1hop` (+0.701, rank 8).
Unlike the public split (where influence fractions are near-zero), both 1-hop and 2-hop influence
fractions carry real weight in the random split on the full test set — the more dispersed random
training labels weaken the distance gradient and allow the Jacobian-weighted composition to become
informative.  The structural features `min_dist_to_same_class_train` and `purity_2hop` still anchor
the prediction, but the influence fractions add independent signal above them.

**`purity_1hop` has a negative coefficient without embeddings**, seemingly contradicting
its expected direction.  This is a collinearity artifact: `purity_2hop` (+1.041) and
`same_train_ratio_1hop` are already capturing the 1-hop same-class signal.
`purity_1hop` then represents the residual — nodes with unexpectedly high 1-hop purity given
their 2-hop composition and training ratios — which may actually correlate with edge cases where
the 1-hop ring is clean but the wider neighbourhood is not, leaving the node structurally
isolated.  Interpret `purity_1hop` jointly with `purity_2hop` rather than independently.

**`mean_cosine_sim_1hop` is near-zero and positive (+0.09) without embeddings**, in contrast to
its consistently negative sign in Cora/PubMed.  Once the structural features are highly
informative (as they are for CiteSeer), cosine_sim carries little residual discriminating power
and its coefficient becomes unstable — consistent with the interaction test finding that
cosine_sim adds negligible information beyond purity.

Note: `avg_spl_to_same_class_train` shows a positive coefficient (+0.32) in both CiteSeer runs,
opposite to its usually negative direction.  This is a collinearity artifact with
`min_dist_to_same_class_train` (−1.36 / −1.24), which dominates the same-class proximity signal.
The conditional effect of average SPL given minimum distance can flip sign.  Interpret these two
SPL features jointly rather than independently.

### 5.11 Influence fraction outperforms raw influence count as a predictor

`same_train_infl_frac_1hop` is the top predictor in the Cora public without-embeddings full-set
run (n=915, coef +1.267) — the fraction of total Jacobian-L1 influence attributable to same-class
training nodes at hop 1 outranks every structural and raw influence feature.  This is a strictly more
informative quantity than either `n_same_train_1hop` (count) or `same_train_ratio_1hop` (fraction
of ring), because it incorporates the *relative Jacobian weight* each training node carries given
the full neighbourhood's degree structure.

**Why the fraction matters more than the count:**  A node can have several same-class training
neighbours yet still be misclassified if those neighbours are high-degree hubs (each contributing
low per-edge influence due to degree normalisation) while one low-degree diff-class training
neighbour contributes a disproportionately large share of the influence budget.
`same_train_infl_frac_1hop` captures this imbalance directly; raw counts and ratios do not.

**Adding the new features improves all three metrics on Cora public (no embeddings):**
AUROC 0.911→0.913, PR-AUC 0.741→0.756, Lift@50 4.23×→4.53×.  The top-50 flagged nodes are now
misclassified at 92% vs the 20.3% baseline — the influence fraction features tighten the ranking.

**The sign reversals for related features are a collinearity consequence**, not contradictions:
- `same_train_ratio_1hop` (−0.301 / −0.515) and `total_infl_same_1hop` (−0.526 / −0.504) flip
  negative in both runs while `same_train_infl_frac_1hop` is strongly positive — all three measure
  same-class training presence at hop 1 from different angles.  Once the LR uses the fraction to
  capture the influence-balance signal, the ratio and raw total represent residuals that can
  correlate with failure in edge cases (a node with many high-degree same-class training neighbours,
  all contributing diluted influence, while sitting in a heterophilic wider neighbourhood).
- `n_diff_train_1hop` is near-zero or slightly positive in both runs — when same-class influence
  fraction is controlled, having diff-class training neighbours indicates structural centrality
  (proximity to labelled nodes generally), a mild success signal.

**`diff_train_infl_frac_1hop` (−0.581 / −0.622) and `diff_train_infl_frac_2hop` (−0.277 / −0.257)**
have the expected negative direction in both runs — nodes where diff-class training nodes command
a large share of the total Jacobian budget at either hop are systematically misclassified.

**`degree` rises to 3rd place (+0.786) in the no-embedding full-set run**, up from near-bottom in
earlier runs without influence features.  Once `same_train_infl_frac_1hop` absorbs the
quality-of-signal dimension (which class dominates the influence budget), degree captures a residual
quantity-of-connections effect: more neighbours → larger total influence budget → more likely
correct, holding the fraction constant.

**`mean_cosine_sim_1hop` collapses to near-zero (+0.0005) in the no-embedding run** — completely
absorbed by the influence fraction features.  When the Jacobian-weighted balance of same vs
diff-class training signal is in the model, raw feature similarity adds nothing.

**At hop 2**, same-class influence is near-zero or small in Cora public — 2-hop same-class
influence carries negligible additional predictive power once hop-1 balance is controlled.  The
2-hop diff-class features (`total_infl_diff_2hop` −0.230/−0.610, `diff_train_infl_frac_2hop`
−0.277/−0.257) retain meaningful signal, suggesting 2-hop diff-class pressure contributes to
failure even when the hop-1 influence balance is relatively favourable.  In Cora random, however,
`same_train_infl_frac_2hop` (+0.573) is the 9th strongest predictor — 2-hop same-class training
influence also carries protective signal in the random split, where the fixed public training nodes
are replaced by a more dispersed labelling.

**Sign anomalies in the Cora random run** follow the same collinearity pattern as Cora public,
but with additional reversals driven by the dominance of `diff_train_infl_frac_1hop`:
- `n_diff_train_1hop` (+0.375) and `diff_train_ratio_1hop` (+0.286) flip positive — once the
  fraction captures the influence imbalance, count and ratio of diff-class training neighbours
  indicate structural centrality rather than failure risk.
- `total_infl_diff_1hop` (+0.252) and `total_infl_same_1hop` (−0.579) both have counterintuitive
  signs — the fraction features absorb the failure signal; the raw totals become residuals.
- `min_dist_to_train` (−0.579) is negative alongside `min_dist_to_same_class_train` (−0.711) —
  collinearity between the two distance features; being close to any training node is captured
  partially by both.

**`diff_train_infl_frac_1hop` (−1.918) is the top predictor for Cora random** on the full test
set without embeddings, confirming it is the most stable and discriminating single feature in
this condition.

**`same_train_infl_frac_2hop` (+1.034) ranks above `same_train_infl_frac_1hop` (+0.923) on the
full test set**, because more than half of nodes have purity_1hop = 1.0 — their 1-hop ring is
entirely same-class, so `same_train_infl_frac_1hop` is uniformly high and less discriminating.
The 2-hop fraction captures more variation across the full population.

**`total_infl_diff_1hop` (−0.905) and `total_infl_diff_2hop` (−0.724)** both rank strongly on
the full test set even alongside `diff_train_infl_frac_1hop` — the raw magnitude of diff-class
influence at both hops adds information beyond the fraction alone, suggesting that absolute
exposure to wrong-class signal matters independently of its relative share of the budget.

**`mean_cosine_sim_1hop` retains a meaningful negative coefficient (−0.277) on the full test
set** — cosine similarity carries residual predictive power when no embedding features are
present to absorb the raw-feature signal.

**Influence fraction features are weak for CiteSeer** — in stark contrast to Cora.  CiteSeer's
lower homophily and larger graph (3327 nodes, 120 training nodes total) make training signal sparse
and unevenly distributed.  For many nodes, the absolute *distance* to the nearest same-class
training node is the binding constraint — whether a same-class or diff-class training node wins
the influence competition matters less when the competition is weak (both parties contributing
small fractions of a thin total budget).  This explains why `total_infl_diff_2hop` — an absolute
magnitude, not a fraction — is a top predictor for CiteSeer: raw 2-hop diff-class exposure matters
more than its share of the budget.

**CiteSeer random vs public split — influence features matter more in the random split.**
In the public split, `min_dist_to_same_class_train` (−1.661) dominates and influence fraction
features are near-zero.  In the random split, `total_infl_diff_2hop` (−1.230) and
`min_dist_to_same_class_train` (−1.178) are nearly tied at the top, and `same_train_infl_frac_2hop`
(+0.751) ranks 3rd.  The mechanism: the public split places 20 fixed training nodes per class in
structured positions, creating strong distance-based gradients.  The random split disperses training
nodes more evenly, weakening the distance signal and allowing influence-weighted composition at
2-hop to become more informative.  On the full test set, `same_train_infl_frac_1hop` (+0.701)
ranks 8th for CiteSeer random — the 1-hop fraction is informative across the full population.

**PubMed random without embeddings: purity_2hop anchors; influence fractions add noise.**
`purity_2hop` (+0.818, rank 1) retains the top position, consistent with the old feature set.
The SPL pair `avg_spl_to_train` (+0.786) and `avg_spl_to_same_class_train` (−0.663) occupies
ranks 2–3 with collinearity-flipped sign on the all-class distance.  `total_infl_diff_1hop`
(−0.442, rank 9) and `total_infl_diff_2hop` (−0.383, rank 11) are correctly negative and carry
moderate weight — raw diff-class influence magnitude retains signal even when fractions do not.
However, influence fraction features are again weak or sign-reversed: `diff_train_infl_frac_2hop`
(+0.315, rank 13) is wrong-signed, `same_train_infl_frac_2hop` (−0.083) and
`total_infl_same_2hop` (−0.087) are also wrong-signed, and all fraction features rank below 13.
AUROC, PR-AUC, and Lift@50 all decreased marginally versus the old feature set (0.861→0.859,
0.606→0.597, 3.68×→3.49×), confirming that the influence fractions hurt slightly rather than
helping.  The PubMed thin-signal regime consistently prevents fraction-based features from being
useful: both splits and both embedding conditions show the same pattern.

**PubMed public without embeddings: influence fractions are weak or sign-reversed; degree dominates.**
AUROC improved modestly from 0.799 (old features) to 0.810, but the new influence features did not
drive the improvement — `degree` (+1.294, rank 1) and `purity_2hop` (+0.789, rank 2) remain the
anchors.  The sign-reversal pattern for same-class influence features mirrors the with-embeddings
run: `total_infl_same_2hop` (−0.490), `same_train_infl_frac_2hop` (−0.461), and
`total_infl_same_1hop` (−0.368) all flip negative, while `total_infl_diff_2hop` (−0.393) and
`diff_train_infl_frac_1hop` (−0.327) are correctly negative.  Once degree absorbs the
quantity-of-connections signal and `purity_2hop` captures neighbourhood composition, the same-class
influence totals become residuals correlated with high-degree nodes in heterogeneous zones — the
sign flips because high 2-hop same-class influence can accompany high 2-hop diff-class influence
in large PubMed neighbourhoods where training signal is mixed.  This is the clearest case across
all conditions where influence fractions fail to add value: the training signal is too thin and the
neighbourhood-size variance too large for Jacobian budget shares to discriminate correctness.

**CiteSeer public purity_1hop interaction test: cosine_sim adds nothing** (49.6% vs 49.4% in the
low-purity group — essentially 0pp gap).  This is the cleanest demonstration across all datasets
that purity alone can be sufficient: once a CiteSeer node has any cross-class 1-hop neighbour,
its raw feature similarity to those neighbours is irrelevant to predicting failure.

---

## 6. Interaction Test: High Cosine Similarity × Low Purity

For every run, nodes are median-split on `mean_cosine_sim_1hop` and on `purity_1hop` / `purity_2hop`
independently, creating four quadrants.  The hypothesis is that the (high_sim, low_purity) quadrant
has the highest misclassification rate.

### 6.1 Results (using purity_1hop)

Note: purity_1hop median = 1.000 on all datasets — more than half of test nodes have a perfectly
homophilic 1-hop neighbourhood.  "Low purity" here means any node with at least one cross-class
1-hop neighbour.

| Dataset | Split | high_sim + low_pur | high_sim + high_pur | low_sim + low_pur | low_sim + high_pur | χ² | p |
|---|---|---|---|---|---|---|---|
| Cora | public | **43.2%** | 7.1% | 37.6% | 9.1% | 55.6 | 0.000 |
| Cora | random | **46.6%** | 8.5% | 40.5% | 5.5% | 77.7 | 0.000 |
| CiteSeer | random | **53.7%** | 9.7% | 52.7% | 9.3% | 46.3 | 0.000 |
| CiteSeer | public | **49.6%** | 12.4% | 49.4% | 7.2% | 38.2 | 0.000 |
| PubMed | random | **41.2%** | 14.2% | 30.6% | 9.9% | 51.9 | 0.000 |
| PubMed | public | **37.0%** | 17.4% | 37.2% | 13.8% | 22.2 | 0.000 |

### 6.2 Results (using purity_2hop)

purity_2hop median varies by dataset and split: 0.843 for Cora public, 0.854 for Cora random,
0.789 for CiteSeer random, 0.800 for CiteSeer public, 0.834 for PubMed random, 0.859 for PubMed
public.  All are sub-1.0 — the median split is structurally informative in every case, with the
high-purity half having a genuinely clean 2-hop ring.  See §6.4 for implications.

| Dataset | Split | high_sim + low_pur | high_sim + high_pur | low_sim + low_pur | low_sim + high_pur | χ² | p |
|---|---|---|---|---|---|---|---|
| Cora | public | **34.6%** | 5.6% | 33.7% | 7.7% | 32.8 | 0.000 |
| Cora | random | **39.2%** | 5.2% | 35.7% | 3.5% | 60.0 | 0.000 |
| CiteSeer | random | **55.3%** | 1.8% | **56.5%** | 5.0% | 65.9 | 0.000 |
| CiteSeer | public | **56.4%** | 3.3% | 52.4% | 2.5% | 77.8 | 0.000 |
| PubMed | random | **41.4%** | 8.0% | 32.6% | 2.9% | 75.3 | 0.000 |
| PubMed | public | 34.5% | 14.7% | **37.8%** | 6.4% | 22.1 | 0.000 |

### 6.3 Base rates

Two base rates are relevant:

- **LR / lift@k base rate** — the misclassification rate within the subset the LR fits on (e.g.,
  39.9% for the CiteSeer mixed-neighbourhood subset of 208 nodes).  Lift@k compares precision@k
  against this rate: Lift@50 = 2.05× means the top-50 flagged nodes are misclassified at 82%
  vs the 39.9% expected from a random draw from the same subset.
- **Interaction test base rate** — the overall test-set misclassification rate (e.g., 29.6% for
  CiteSeer, 196/663), since the quadrant counts sum to the full test set.  Each quadrant's
  misclassification rate should be compared against this reference to quantify enrichment.

### 6.4 What the quadrant cross-tabulation reveals

The cross-tabulation independently varies two dimensions — neighbourhood class composition (purity)
and raw feature similarity (cosine_sim) — to test whether they interact or act independently on
failure rate.

**Necessity and sufficiency of each factor:**

- If misclassification rates are similar across cosine_sim groups *within* each purity group (as
  in CiteSeer × purity_2hop), purity is sufficient on its own — knowing purity alone predicts
  outcome; adding cosine_sim provides negligible extra information.
- If high_sim + low_purity fails substantially more than low_sim + low_purity, cosine_sim
  amplifies the purity effect — the two factors interact, and both are needed to characterise
  the highest-risk nodes.  This is the pattern on Cora and PubMed.

**CiteSeer × purity_2hop: purity is near-sufficient.**

| | low purity_2hop | high purity_2hop |
|---|---|---|
| high cosine_sim | 56.4% | 3.3% |
| low cosine_sim | 52.4% | 2.5% |

The cosine_sim rows are nearly identical within each purity column.  Purity_2hop alone determines
the outcome: high 2-hop purity → almost never fails (2.5–3.3%); low 2-hop purity → fails more
than half the time regardless of feature similarity.  This is because CiteSeer's purity_2hop
median of 0.800 creates a genuinely meaningful structural split — the high-purity half has a
clean 2-hop ring, the low-purity half has a substantially mixed one.  For CiteSeer public purity_1hop, the cosine_sim interaction is completely absent: 49.6% vs 49.4%
within the low-purity group — once a CiteSeer node has any cross-class 1-hop neighbour, feature
similarity is irrelevant to predicting failure.  On Cora (purity_2hop median 0.843–0.854), the
split is less extreme, so cosine_sim picks up residual variance and the interaction reappears
(39.2% vs 35.7%).

**Cora / PubMed: cosine_sim amplifies the purity effect.**

When purity is not a clean separator, a node whose raw features resemble its (wrong-class)
neighbours has even less signal available to override the heterogeneous neighbourhood aggregate.
High cosine_sim tightens the GCN's embedding toward its neighbours' representation — helpful when
those neighbours are same-class, harmful when they are not.  This is the interaction: low purity
creates the structural condition for failure; high cosine_sim worsens it by reducing the model's
ability to separate the focal node from its neighbourhood.

**Practical implication:** The quadrants define a triage hierarchy.  Across all datasets:
- (high purity) → safe zone: 1.8–14.7% misclassification.
- (low purity, any cosine_sim) → danger zone: 30–57% misclassification.
- (low purity + high cosine_sim) → highest-risk in homophilic graphs where purity alone is
  insufficient.

The χ² statistic confirms significance in all cases but should not be compared directly across
datasets — it is driven by sample size and class balance.  The misclassification rate contrast
within each table is the interpretable quantity.

The effect is strongest on Cora random using purity_1hop (χ²=77.7) and weakest on PubMed public
(χ²≈22).  This mirrors the overall AUROC pattern: Cora failure is more structurally determined;
PubMed public failure is less so.

**Interpretation of the interaction:** Low purity creates the structural condition for failure —
the neighbourhood is class-heterogeneous.  High raw feature cosine similarity amplifies this: if
the focal node's raw features resemble its (wrong-class) neighbours, the GCN has even less signal
to distinguish the node from its neighbourhood.  The interaction is a necessary but not sufficient
condition for failure — not every (high_sim, low_purity) node fails, but this quadrant
concentrates failures at 2–4× the overall rate.

---

## 7. Publishable Summary

1. **GCN misclassification is systematic, not random** — PR-AUC 0.59–0.88 (3–4× the baseline
   rate) from a simple logistic regression on neighbourhood structure, training signal proximity,
   and feature-space similarity.  Failures concentrate among structurally identifiable nodes: the
   top-50 flagged nodes are misclassified at 4–5× the baseline rate.  AUROC 0.80–0.94 confirms
   strong overall discrimination for cross-dataset comparison.

2. **Structural predictability varies with dataset size and homophily.**  For cross-dataset
   comparison (AUROC, prevalence-independent): Cora ~0.91 > CiteSeer ~0.88 > PubMed random ~0.86
   > PubMed public ~0.80.  When structural features are insufficient, embedding-space purity
   recovers the gap (PubMed public: PR-AUC 0.591 → 0.697 with embeddings).  CiteSeer's lower
   homophily (~74%) makes `purity_2hop` highly discriminating: high 2-hop purity → 1.8%
   misclassification; low 2-hop purity → 55–57%.

3. **The dominant structural predictors, consistent across all conditions:**
   - 2-hop neighbourhood purity (`purity_2hop`)
   - Proximity to same-class training nodes (`min_dist_to_same_class_train`, `avg_spl_to_same_class_train`)
   - Presence of diff-class training nodes in the 1-hop ring (`diff_train_ratio_1hop`)

4. **Degree is a proxy for structural advantage, not a causal driver** — its importance is
   inversely proportional to the availability of better signal-quality features.  In dense,
   homophilic graphs with nearby training nodes (Cora), influence fraction features absorb degree's
   proxy role and it drops to rank 14.  In large, thin-signal graphs (PubMed public, 20 training
   nodes across ~19K nodes), no fraction-based feature can discriminate reliably and degree rises
   to rank 1 (+1.294) as the best available quantity-of-connections proxy.  High degree is not
   uniformly protective: in low-purity neighbourhoods it accumulates many wrong-class signals,
   producing a confidently wrong aggregate.  Low-degree nodes are not uniformly disadvantaged:
   their 2-hop neighbourhood can expand through high-degree 1-hop neighbours, and the genuine
   risk factors (peripheral position, single wrong-class direct neighbour) are captured by SPL
   and purity features rather than degree itself.

5. **High raw feature cosine similarity combined with low neighbourhood purity is the canonical
   failure quadrant** (χ² p ≈ 0, all datasets and splits), with misclassification rates 2–4×
   the baseline.  This is a necessary-but-not-sufficient structural signature of GCN failure.

6. **Embedding-space purity** (whether the GCN successfully separates same-class from diff-class
   neighbours in representation space) is a candidate decisive factor for structurally ambiguous
   nodes — see `RESULTS_EMBEDDINGS.md` for preliminary evidence pending methodology validation.
