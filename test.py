import torch


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    results = {}
    for split, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        results[split] = correct / mask.sum().item()
    return results
