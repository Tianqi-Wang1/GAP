import os
import torch
from tqdm import tqdm


@torch.no_grad()
def extract_image_features(data_loader, image_encoder, device, dtype, normalize=True):
    image_encoder.eval()

    all_feats = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Extracting image features"):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        try:
            feats = image_encoder(images.type(dtype))
        except Exception:
            feats = image_encoder(images.float())

        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.float()

        all_feats.append(feats.cpu())
        all_labels.append(labels.cpu())

    all_feats = torch.cat(all_feats, dim=0)   # [N, D]
    all_labels = torch.cat(all_labels, dim=0) # [N]

    return all_feats, all_labels


def save_feature_cache(save_path, features, labels):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "features": features,
        "labels": labels
    }, save_path)


def load_feature_cache(save_path):
    data = torch.load(save_path, map_location="cpu")
    return data["features"], data["labels"]