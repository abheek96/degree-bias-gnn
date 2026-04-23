import logging
import os
import random

import numpy as np
import torch
from torch_geometric.utils import degree, to_dense_adj

log = logging.getLogger(__name__)


def load_or_create_split(dataset_cfg, split: str, seed: int,
                         cache_dir: str = "dataset_cache"):
    """Return a split Data object, loading from disk cache if available.

    Cache path: ``{cache_dir}/{name}_{split}_{CC|noCC}_seed{seed}.pt``

    For public+noCC the split is deterministic, but we still key by seed so
    the call signature is uniform. Both ``main.py`` and analysis scripts can
    call this with the same arguments and are guaranteed to get the same
    train/val/test masks.
    """
    from dataset import load_dataset

    cc_tag = "CC" if dataset_cfg.get("use_cc", False) else "noCC"
    name   = dataset_cfg["name"]
    fname  = f"{name}_{split}_{cc_tag}_seed{seed}.pt"
    path   = os.path.join(cache_dir, fname)

    if os.path.exists(path):
        log.info("Loading cached split: %s", path)
        return torch.load(path, weights_only=False)

    log.info("No cache found — creating split and saving to %s", path)
    _set_seed(seed)
    data = load_dataset(dataset_cfg)
    if split == "random":
        pass  # seed already set; apply_split will draw from current RNG state
    data = apply_split(data, split, dataset_cfg)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(data, path)
    log.info("Cached split saved: %s", path)
    return data


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_split(data, split, dataset_cfg):
    """Apply or validate train/val/test masks.

    - public + use_cc=True : PyG's LargestConnectedComponents already filters
      the masks in-place; this function logs the surviving counts and then tops
      up any class whose training count fell below num_train_per_class by
      sampling additional free nodes from the connected component.
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

            # Restore num_train_per_class balance if CC filtering dropped nodes
            num_train_per_class = dataset_cfg.get("num_train_per_class", 20)
            num_classes = int(data.y.max().item()) + 1
            train_mask = data.train_mask.clone()
            occupied   = data.train_mask | data.val_mask | data.test_mask
            any_topped_up = False

            for c in range(num_classes):
                class_train = (train_mask & (data.y == c)).nonzero(as_tuple=False).view(-1)
                have    = class_train.size(0)
                deficit = num_train_per_class - have
                if deficit <= 0:
                    continue

                # Free nodes: class c, inside the CC, not in any mask
                class_all  = (data.y == c).nonzero(as_tuple=False).view(-1)
                free       = class_all[~occupied[class_all]]

                if free.size(0) == 0:
                    log.warning(
                        "Class %d: deficit=%d but no free nodes available — "
                        "cannot restore balance", c, deficit)
                    continue

                if free.size(0) < deficit:
                    log.warning(
                        "Class %d: deficit=%d but only %d free nodes — "
                        "partial top-up", c, deficit, free.size(0))
                    deficit = free.size(0)

                new_nodes = free[torch.randperm(free.size(0))[:deficit]]
                train_mask[new_nodes] = True
                occupied[new_nodes]   = True
                any_topped_up = True
                log.info("Class %d: topped up %d node(s) (was %d, now %d)",
                         c, deficit, have, have + deficit)

            if any_topped_up:
                data = data.clone()
                data.train_mask = train_mask
                log.info("Public split after top-up: %d train  %d val  %d test",
                         data.train_mask.sum().item(),
                         data.val_mask.sum().item(),
                         data.test_mask.sum().item())

        return data

    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1
    num_train_per_class = dataset_cfg.get("num_train_per_class", 20)
    num_val = dataset_cfg.get("num_val", 500)

    # Match num_test to the public split when available.
    #
    # Datasets like Planetoid (Cora, CiteSeer, PubMed) ship with a built-in
    # test_mask.  If use_cc=True, LargestConnectedComponents has already
    # filtered that mask in-place, so its sum reflects the post-CC test count
    # (e.g. 915 for Cora+CC instead of the nominal 1 000).  Using that same
    # count for the random split keeps the two split types on equal footing:
    # both evaluate on the same number of test nodes, making accuracy numbers
    # directly comparable.
    #
    # For datasets without a built-in 1-D test mask (Amazon, Coauthor, …) the
    # condition is False and we fall back to the value in config.yaml.
    if (hasattr(data, "test_mask")
            and data.test_mask is not None
            and data.test_mask.dim() == 1):
        num_test = int(data.test_mask.sum().item())
        log.info("num_test set to %d to match public split (post-CC if applicable)",
                 num_test)
    else:
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



# ── Unused functions ──────────────────────────────────────────────────────────

#
# def get_str_info(adj, hop):
#     """Return the total number of length-``hop`` walks originating from each node.
#
#     Raises the dense adjacency matrix to the power ``hop`` via
#     ``np.linalg.matrix_power``.  Entry [i, j] of adj^k counts walks of
#     exactly length k from i to j; summing over j gives a per-node scalar.
#
#     Note: counts *walks*, not distinct paths — backtracking is allowed, so
#     e.g. i→k→i is a valid length-2 walk.  For hop=1 this equals the node
#     degree; for hop=2 the diagonal contribution is the degree itself
#     (one backtrack per neighbour).
#
#     Cost: O(N² ) memory, O(N³ log hop) compute — expensive for large graphs.
#     """
#     str_info = np.linalg.matrix_power(adj, hop)
#     return np.sum(str_info, axis=1)
#
# def make_deg_groups(data, n_groups=2):
#     """Compute degree-based node groups and structural neighbourhood info.
#
#     Adds the following attributes to ``data`` in-place:
#         deg        : raw degree per node (float tensor)
#         deg_labels : degree normalised to [0, 1] by the maximum degree
#         deg_group  : integer group index in {0, …, n_groups-1} via
#                      equal-width degree bins
#         group1     : 1-hop structural info from get_str_info
#         group2     : 2-hop structural info from get_str_info
#
#     Parameters
#     ----------
#     data     : PyG Data object (original graph or CC subgraph)
#     n_groups : number of equal-width degree buckets
#
#     Returns
#     -------
#     data with the above attributes added.
#     """
#     deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
#     max_deg = deg.max().item()
#
#     data.deg = deg
#     data.deg_labels = deg / max_deg
#
#     # Equal-width bins; bucketize assigns each node a group in [0, n_groups)
#     boundaries = torch.linspace(0, max_deg + 1, n_groups + 1)
#     data.deg_group = torch.bucketize(deg, boundaries[1:-1])
#
#     # max_num_nodes prevents over-allocation after CC filtering reindexes nodes
#     adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).cpu().numpy()
#     data.group1 = get_str_info(adj, hop=1)
#     data.group2 = get_str_info(adj, hop=2)
#
#     return data
