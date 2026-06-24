import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class LinearProbe(nn.Module):
    def __init__(self, feat_dim, num_classes, bias=False):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def compute_probe_accuracy(model, feats, labels, batch_size=1024, device="cuda"):
    model.eval()
    total = 0
    correct = 0

    for start in range(0, feats.size(0), batch_size):
        end = start + batch_size
        x = feats[start:end].to(device)
        y = labels[start:end].to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

    return 100.0 * correct / total


def fit_surrogate_probe(
    train_feats,
    train_labels,
    num_classes,
    lr=1e-2,
    weight_decay=1e-4,
    epochs=100,
    batch_size=256,
    device="cuda",
    bias=False,
    init_mode="prototype",   # "random" or "prototype"
):
    train_feats = train_feats.float()
    train_labels = train_labels.long()

    feat_dim = train_feats.size(1)
    model = LinearProbe(feat_dim, num_classes, bias=bias).to(device)
    model = model.float()

    if init_mode == "prototype":
        proto = compute_visual_prototypes(
            train_feats=train_feats.to(device),
            train_labels=train_labels.to(device),
            num_classes=num_classes,
            normalize=True
        )   # [K, D]

        with torch.no_grad():
            model.fc.weight.copy_(proto)
            if bias and model.fc.bias is not None:
                model.fc.bias.zero_()

        print(">> Probe initialized from visual prototypes")
    else:
        print(">> Probe uses random initialization")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    dataset = TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0

        for x, y in loader:
            x = x.float().to(device)
            y = y.long().to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            total += y.size(0)

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            avg_loss = running_loss / total
            print(f">> Probe epoch [{epoch+1}/{epochs}] loss={avg_loss:.6f}")

    train_acc = compute_probe_accuracy(model, train_feats, train_labels, device=device)
    print(f">> Probe train acc: {train_acc:.2f}%")

    probe_weight = model.fc.weight.detach().cpu()   # [K, D]
    return probe_weight, train_acc


def save_probe_cache(save_path, probe_weight, train_acc=None, meta=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "probe_weight": probe_weight,   # [K, D]
        "train_acc": train_acc,
        "meta": meta if meta is not None else {}
    }, save_path)


def load_probe_cache(save_path):
    data = torch.load(save_path, map_location="cpu")
    return data["probe_weight"], data.get("train_acc", None), data.get("meta", {})


import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_visual_prototypes(train_feats, train_labels, num_classes, normalize=True):
    """
    train_feats: [N, D]
    train_labels: [N]
    return:
        prototypes: [K, D]
    """
    train_feats = train_feats.float()
    train_labels = train_labels.long()

    N, D = train_feats.shape
    K = num_classes

    prototypes = torch.zeros(K, D, dtype=train_feats.dtype, device=train_feats.device)
    counts = torch.zeros(K, dtype=train_feats.dtype, device=train_feats.device)

    prototypes.index_add_(0, train_labels, train_feats)
    counts.index_add_(0, train_labels, torch.ones_like(train_labels, dtype=train_feats.dtype))

    counts = counts.clamp_min(1.0).unsqueeze(1)   # [K, 1]
    prototypes = prototypes / counts

    if normalize:
        prototypes = F.normalize(prototypes, dim=1)

    return prototypes

@torch.no_grad()
def solve_ridge_init_G(base_text_features, B_T, ref_weight, ridge_lambda=1e-3):
    """
    base_text_features: [K, D] = T
    B_T: [D, r]
    ref_weight: [K, D] = W_ref, e.g. probe_weight or visual prototypes

    return:
        G_init: [r, D]
    """
    T = base_text_features.float()
    B_T = B_T.float()
    W_ref = ref_weight.float()

    # U = T B_T, shape [K, r]
    U = T @ B_T

    # target residual outside text subspace
    W_tar = W_ref - W_ref @ B_T @ B_T.t()   # [K, D]

    r = U.shape[1]
    I = torch.eye(r, device=U.device, dtype=U.dtype)

    # G = (U^T U + λI)^(-1) U^T W_tar
    G_init = torch.linalg.solve(U.t() @ U + ridge_lambda * I, U.t() @ W_tar)

    return G_init