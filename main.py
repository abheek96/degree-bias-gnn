import argparse
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from train import train
from test import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run(cfg, run_id, device):
    from dataset import load_dataset
    from models import get_model

    data = load_dataset(cfg["dataset"])
    data = data.to(device)

    model_cfg = cfg["model"]
    model = get_model(
        model_cfg["name"],
        in_dim=data.num_node_features,
        hidden_dim=model_cfg["hidden_dim"],
        out_dim=int(data.y.max().item()) + 1,
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    train_cfg = cfg["train"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0
    patience = train_cfg.get("patience", 0)

    epoch_bar = tqdm(range(1, train_cfg["epochs"] + 1), desc=f"Run {run_id}", leave=False)
    for epoch in epoch_bar:
        loss = train(model, data, optimizer, criterion)
        results = evaluate(model, data)

        if results["val"] > best_val_acc:
            best_val_acc = results["val"]
            best_test_acc = results["test"]
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_bar.set_postfix(loss=f"{loss:.4f}", val=f"{results['val']:.4f}", test=f"{results['test']:.4f}")

        if patience > 0 and patience_counter >= patience:
            epoch_bar.close()
            tqdm.write(f"  [Run {run_id}] Early stopping at epoch {epoch}")
            break

    return best_val_acc, best_test_acc


def main():
    parser = argparse.ArgumentParser(description="Graph Learning Experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    base_seed = cfg.get("seed", 42)
    num_runs = cfg.get("num_runs", 1)

    val_accs, test_accs = [], []

    for i in tqdm(range(1, num_runs + 1), desc="Runs"):
        seed = base_seed + i - 1
        set_seed(seed)
        tqdm.write(f"\n=== Run {i}/{num_runs} (seed={seed}) ===")
        val_acc, test_acc = run(cfg, i, device)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        tqdm.write(f"  Best Val: {val_acc:.4f}  Test: {test_acc:.4f}")

    val_mean, val_std = np.mean(val_accs), np.std(val_accs)
    test_mean, test_std = np.mean(test_accs), np.std(test_accs)
    print(f"\n{'='*50}")
    print(f"Results over {num_runs} runs:")
    print(f"  Val:  {val_mean:.4f} +/- {val_std:.4f}")
    print(f"  Test: {test_mean:.4f} +/- {test_std:.4f}")


if __name__ == "__main__":
    main()
