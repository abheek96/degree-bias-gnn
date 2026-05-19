# Node CSV — Column Definitions

| Column | Description |
|---|---|
| `node_idx` | 0-based node index in the graph (PyTorch Geometric convention) |
| `true_label` | Ground-truth class label |
| `is_train` | 1 if training node, 0 otherwise |
| `is_val` | 1 if validation node, 0 otherwise |
| `is_test` | 1 if test node, 0 otherwise. Nodes with all three = 0 are unassigned |
| `degree` | Number of direct (1-hop) neighbours |
| `khop_degree_k2` | Number of distinct nodes reachable within 2 hops — the receptive field of a 2-layer GNN |
| `dist_to_train` | Minimum hop-distance to the nearest training node of any class. 0 for training nodes. NaN if unreachable within 10 hops |
| `dist_to_same_class_train` | Minimum hop-distance to the nearest same-class training node. 0 for training nodes of that class. NaN if unreachable |
| `avg_spl_to_train` | Average shortest path length to all reachable training nodes |
| `avg_spl_to_same_class_train` | Average shortest path length to all reachable same-class training nodes only |
| `has_labelled_neighbour` | 1 if the node has at least one direct training node as a neighbour, 0 otherwise |
| `purity_k1` | Fraction of immediate (1-hop) neighbours sharing the node's true class label |
| `purity_k2` | Fraction of all nodes reachable within 2 hops sharing the node's true class label |
| `delta_purity` | `purity_k2 − purity_k1`. Negative = neighbourhood becomes noisier as receptive field expands |
| `pred_label_mode` | Most frequently predicted class across all runs (mode) |
| `correct_mean` | Fraction of runs in which the node was correctly classified (0.0 = always wrong, 1.0 = always correct) |
| `correct` | 1 if `correct_mean ≥ 0.5` (correct in the majority of runs), 0 otherwise |
