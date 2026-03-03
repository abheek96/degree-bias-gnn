import logging
import numpy as np
import torch
from torch_geometric.utils import degree, to_dense_adj

log = logging.getLogger(__name__)


def apply_split(data, split, dataset_cfg):
    """Apply or validate train/val/test masks.

    - public + use_cc=True : PyG's LargestConnectedComponents already filters
      the masks in-place; this function just logs the surviving counts so the
      impact of CC filtering on each split is visible.
    - public + use_cc=False: no-op.
    - random (any use_cc)  : create masks from scratch using the active RNG
      state — call set_seed() before this to fix the split across all runs.
    """
    use_cc = dataset_cfg.get("use_cc", False)

    if split == "public":
        if use_cc:
            log.info("Public split after CC filter: %d train  %d val  %d test",
                     data.train_mask.sum().item(),
                     data.val_mask.sum().item(),
                     data.test_mask.sum().item())
        return data

    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1
    breakpoint()
    num_train_per_class = dataset_cfg.get("num_train_per_class", 20)
    num_val = dataset_cfg.get("num_val", 500)
    num_test = dataset_cfg.get("num_test", 1000)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.randperm(idx.size(0))[:num_train_per_class]
        train_mask[idx[perm]] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]
    val_mask[remaining[:num_val]] = True
    test_mask[remaining[num_val:num_val + num_test]] = True

    data = data.clone()
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    log.info("Random split: %d train  %d val  %d test",
             train_mask.sum().item(), num_val, num_test)
    return data

def get_str_info(adj, hop):
    """Return the total number of length-``hop`` walks originating from each node.

    Raises the dense adjacency matrix to the power ``hop`` via
    ``np.linalg.matrix_power``.  Entry [i, j] of adj^k counts walks of
    exactly length k from i to j; summing over j gives a per-node scalar.

    Note: counts *walks*, not distinct paths — backtracking is allowed, so
    e.g. i→k→i is a valid length-2 walk.  For hop=1 this equals the node
    degree; for hop=2 the diagonal contribution is the degree itself
    (one backtrack per neighbour).

    Cost: O(N² ) memory, O(N³ log hop) compute — expensive for large graphs.
    """
    str_info = np.linalg.matrix_power(adj, hop)
    return np.sum(str_info, axis=1)

def make_deg_groups(data, n_groups=2):
    """Compute degree-based node groups and structural neighbourhood info.

    Adds the following attributes to ``data`` in-place:
        deg        : raw degree per node (float tensor)
        deg_labels : degree normalised to [0, 1] by the maximum degree
        deg_group  : integer group index in {0, …, n_groups-1} via
                     equal-width degree bins
        group1     : 1-hop structural info from get_str_info
        group2     : 2-hop structural info from get_str_info

    Parameters
    ----------
    data     : PyG Data object (original graph or CC subgraph)
    n_groups : number of equal-width degree buckets

    Returns
    -------
    data with the above attributes added.
    """
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    max_deg = deg.max().item()

    data.deg = deg
    data.deg_labels = deg / max_deg

    # Equal-width bins; bucketize assigns each node a group in [0, n_groups)
    boundaries = torch.linspace(0, max_deg + 1, n_groups + 1)
    data.deg_group = torch.bucketize(deg, boundaries[1:-1])

    # max_num_nodes prevents over-allocation after CC filtering reindexes nodes
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).cpu().numpy()
    data.group1 = get_str_info(adj, hop=1)
    data.group2 = get_str_info(adj, hop=2)

    return data