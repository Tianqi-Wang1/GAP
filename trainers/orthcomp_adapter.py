"""
Minimal trainer for text-conditioned CLIP classifier compensation.

This file intentionally keeps only the current mainline method:
    W = T + beta * G_theta(T B_T)
where T is the frozen CLIP text classifier, B_T is a compact text-coordinate
basis built from T, and G_theta is a shared nonlinear residual generator.

Removed from the exploratory file:
    - class-specific TaskRes variants
    - hard orthogonal residual basis variants
    - visual/prototype dictionary variants
    - probe fitting and feature-cache logic
    - prototype CE and geometry losses

The trainer name is kept as `OrthComp_adapter` for compatibility with existing
DASSL configs/scripts.
"""

import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_pretrained_weights, load_checkpoint

import os
import json
from tqdm import tqdm

from clip import clip
from trainers.imagenet_templates import IMAGENET_TEMPLATES_SELECT


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATE_ENSEMBLES = {
    "OxfordFlowers": [
        "a photo of a {}, a type of flower.",
        "a close-up photo of a {} flower.",
        "a photo of the flower {}.",
        "a macro photo of a {}.",
        "a photo of a blooming {}.",
    ],
    "Food101": [
        "a photo of {}, a type of food.",
        "a close-up photo of {}.",
        "a photo of a plate of {}.",
        "a photo of freshly made {}.",
        "a restaurant photo of {}.",
    ],
    "OxfordPets": [
        "a photo of a {}, a type of pet.",
        "a photo of the pet {}.",
        "a close-up photo of a {}.",
        "a photo of a {} animal.",
        "a portrait photo of a {}.",
    ],
    "FGVCAircraft": [
        "a photo of a {}, a type of aircraft.",
        "a photo of the aircraft {}.",
        "a side view photo of a {}.",
        "a photo of a {} airplane.",
        "a photo of a flying {}.",
    ],
    "StanfordCars": [
        "a photo of a {}.",
        "a photo of the car {}.",
        "a side view photo of a {}.",
        "a photo of a {} vehicle.",
        "a close-up photo of a {}.",
    ],
    "DescribableTextures": [
        "{} texture.",
        "a photo of {} texture.",
        "a close-up photo of {} texture.",
        "a photo showing a {} surface.",
        "a texture image of {}.",
    ],
    "EuroSAT": [
        "a centered satellite photo of {}.",
        "a satellite image of {}.",
        "an aerial photo of {}.",
        "a remote sensing image of {}.",
        "a satellite view of {}.",
    ],
    "SUN397": [
        "a photo of a {}.",
        "a photo of the scene {}.",
        "a photo of a {} scene.",
        "an indoor or outdoor scene of {}.",
        "a view of a {}.",
    ],
    "Caltech101": [
        "a photo of a {}.",
        "a close-up photo of a {}.",
        "a photo of the object {}.",
        "a cropped photo of a {}.",
        "a clean photo of a {}.",
    ],
    "UCF101": [
        "a photo of a person doing {}.",
        "a video frame of a person doing {}.",
        "a photo of the action {}.",
        "a person is performing {}.",
        "a scene of a person doing {}.",
    ],
}

def cfg_get(cfg, path, default=None):
    """Small helper for optional DASSL/YACS config fields."""
    obj = cfg
    try:
        for key in path.split("."):
            obj = getattr(obj, key)
        return obj
    except Exception:
        return default


def get_prompt_templates(cfg):
    dataset = cfg.DATASET.NAME
    use_ensemble = bool(cfg_get(cfg, "TRAINER.OrthComp_adapter.PROMPT_ENSEMBLE", False))

    imagenet_like_datasets = [
        "ImageNet",
        "ImageNetSketch",
        "ImageNetA",
        "ImageNetR",
        "ImageNetV2",
        "ImageNetAFilter",
        "ImageNetRFilter",
    ]

    if dataset in imagenet_like_datasets:
        return IMAGENET_TEMPLATES_SELECT

    if use_ensemble and dataset in CUSTOM_TEMPLATE_ENSEMBLES:
        return CUSTOM_TEMPLATE_ENSEMBLES[dataset]

    if dataset not in CUSTOM_TEMPLATES:
        raise KeyError(f"No handcrafted prompt template for dataset: {dataset}")

    return [CUSTOM_TEMPLATES[dataset]]


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return clip.build_model(state_dict or model.state_dict())


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def get_base_text_features(cfg, classnames, clip_model, text_encoder):
    """
    Build CLIP text classifier features from class names.

    By default, this preserves the previous behavior: prompt features are averaged
    without per-prompt normalization. To test a cleaner CLIP-style variant, set:
        TRAINER.OrthComp_adapter.NORMALIZE_TEXT_FEATURES = True
    """
    original_device = next(text_encoder.parameters()).device

    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

    text_device = next(text_encoder.parameters()).device
    token_device = clip_model.token_embedding.weight.device

    templates = get_prompt_templates(cfg)
    normalize_text = bool(cfg_get(cfg, "TRAINER.OrthComp_adapter.NORMALIZE_TEXT_FEATURES", False))

    print(f">> Use {len(templates)} prompt template(s) for {cfg.DATASET.NAME}")
    for i, template in enumerate(templates):
        print(f"   [{i}] {template}")
    print(f">> Normalize text features before/after prompt averaging: {normalize_text}")

    text_encoder.eval()
    text_features = []

    with torch.no_grad():
        for classname in classnames:
            classname = classname.replace("_", " ")
            prompts = [template.format(classname) for template in templates]

            tokens = clip.tokenize(prompts).to(token_device)
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            embeddings = embeddings.to(text_device)
            tokens = tokens.to(text_device)

            class_features = text_encoder(embeddings, tokens).float()

            if normalize_text:
                class_features = F.normalize(class_features, dim=-1)

            class_feature = class_features.mean(dim=0)

            if normalize_text:
                class_feature = F.normalize(class_feature, dim=0)

            text_features.append(class_feature)

    text_features = torch.stack(text_features, dim=0)
    text_encoder = text_encoder.to(original_device)
    return text_features.to(original_device)


def build_text_basis(text_features, rank=128, rank_proj=128, eps=1e-6, center=False):
    """
    Build a compact orthonormal text-coordinate basis B_T from class text features.

    Args:
        text_features: [K, D]
        rank: maximum basis rank
        eps: numerical rank threshold
        center: whether to center the class text matrix before SVD

    Returns:
        B_T: [D, r]
        singular_values: [min(D, K)]
        selected_rank: int
    """
    T = text_features.float()
    if center:
        T = T - T.mean(dim=0, keepdim=True)

    # SVD on T^T so that left singular vectors live in feature space.
    U, S, _ = torch.linalg.svd(T.t(), full_matrices=False)
    numerical_rank = int((S > eps).sum().item())
    selected_rank = min(max(1, numerical_rank), int(rank), U.size(1))
    selected_rank_Proj = min(max(1, numerical_rank), int(rank_proj), U.size(1))
    B_T = U[:, :selected_rank].contiguous()
    B_T_Proj = U[:, :selected_rank_Proj].contiguous()
    return B_T, B_T_Proj, S, selected_rank

def _get_model_file(epoch):
    if epoch is None or int(epoch) < 0:
        return "model-best.pth.tar"
    return "model.pth.tar-" + str(int(epoch))


def _find_state_key(state_dict, suffix):
    """
    Robustly find a key ending with suffix.
    Example suffix: 'prompt_learner.B_T'
    """
    if suffix in state_dict:
        return suffix

    candidates = [k for k in state_dict.keys() if k.endswith(suffix)]
    if len(candidates) == 0:
        raise KeyError(f"Cannot find key ending with '{suffix}' in checkpoint")
    if len(candidates) > 1:
        print(f">> Warning: multiple keys match {suffix}: {candidates}. Use {candidates[0]}")
    return candidates[0]


def load_source_basis_from_checkpoint(model_dir, epoch=None, model_name="model"):
    """
    Load ImageNet-trained source text basis from checkpoint.

    Expected checkpoint path:
        model_dir/model/model.pth.tar-{epoch}
    or:
        model_dir/model/model-best.pth.tar
    """
    model_file = _get_model_file(epoch)
    model_path = osp.join(model_dir, model_name, model_file)

    if not osp.exists(model_path):
        raise FileNotFoundError(f"Source checkpoint not found: {model_path}")

    checkpoint = load_checkpoint(model_path)
    state_dict = checkpoint["state_dict"]

    key_bt = _find_state_key(state_dict, "prompt_learner.B_T")
    key_bproj = _find_state_key(state_dict, "prompt_learner.B_Proj")

    B_T = state_dict[key_bt].float().contiguous()
    B_Proj = state_dict[key_bproj].float().contiguous()

    print(f">> Loaded source B_T from: {model_path}")
    print(f">> Source B_T shape: {B_T.shape}")
    print(f">> Source B_Proj shape: {B_Proj.shape}")

    return B_T, B_Proj

class TextConditionedCompensator(nn.Module):
    """
    Minimal category-flexible residual generator.

    Form:
        U = T @ B_T
        R = Linear(U) + MLP(U)
        W = T + beta * R

    The default is full-space residual compensation. A projection switch is kept
    only for ablation/debugging:
        PROJECT_RESIDUAL = True => R <- R - (R B_T) B_T^T
    """

    def __init__(self, cfg, base_text_features, B_T, B_Proj):
        super().__init__()
        self.beta = float(cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE)

        self.register_buffer("base_text_features", base_text_features.float())  # [K, D]
        self.register_buffer("B_T", B_T.float())                                # [D, r]
        self.register_buffer("B_Proj", B_Proj.float())

        K, D = base_text_features.shape
        r = B_T.shape[1]

        hidden_dim = int(cfg_get(cfg, "TRAINER.OrthComp_adapter.HIDDEN_DIM", min(D, 4 * r)))
        self.use_layer_norm = bool(cfg_get(cfg, "TRAINER.OrthComp_adapter.USE_LAYER_NORM", False))
        self.project_residual = bool(cfg_get(cfg, "TRAINER.OrthComp_adapter.PROJECT_RESIDUAL", False))

        self.cond_norm = nn.LayerNorm(r) if self.use_layer_norm else nn.Identity()

        self.linear = nn.Linear(r, D, bias=False)
        nn.init.zeros_(self.linear.weight)

        self.delta = nn.Sequential(
            nn.Linear(r, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D, bias=False),
        )
        nn.init.zeros_(self.delta[-1].weight)

        print(
            f">> TextConditionedCompensator: K={K}, D={D}, r={r}, "
            f"hidden={hidden_dim}, beta={self.beta}, "
            f"layer_norm={self.use_layer_norm}, project={self.project_residual}"
        )

    def get_base_text_features(self):
        return self.base_text_features

    def compute_raw_residual(self):
        """
        Compute the raw generated residual before optional projection.

        Returns:
            U: [K, r], text coordinates
            R_raw: [K, D], raw residual
        """
        T = self.base_text_features                      # [K, D]
        U = self.cond_norm(T @ self.B_T)                 # [K, r]
        R_raw = self.linear(U) + self.delta(U)           # [K, D]
        return U, R_raw

    def compute_residual(self):
        """
        Compute the effective residual used by the classifier.

        If PROJECT_RESIDUAL=True:
            R_eff = R_raw - (R_raw @ B_Proj) @ B_Proj^T
        Otherwise:
            R_eff = R_raw
        """
        U, R = self.compute_raw_residual()

        if self.project_residual:
            R = R - (R @ self.B_Proj) @ self.B_Proj.t()

        return U, R

    @torch.no_grad()
    def residual_energy_analysis(self, top_rank=32, center=False, svd_eps=1e-6, eps=1e-12):
        """
        Analyze how much residual energy lies in:
            1. the full text subspace;
            2. the top-k text subspace.

        We report both:
            - raw residual: before optional projection;
            - effective residual: after optional projection.

        This is useful because when PROJECT_RESIDUAL=True, the effective
        residual should have nearly zero energy in the projected top-k subspace
        by construction, while the raw residual tells us what the generator
        originally attempted to produce.
        """
        self.eval()

        T = self.base_text_features.float()              # [K, D]

        if center:
            T_for_svd = T - T.mean(dim=0, keepdim=True)
        else:
            T_for_svd = T

        # SVD on T^T so that U lives in feature space: [D, rank]
        U_svd, S, _ = torch.linalg.svd(T_for_svd.t(), full_matrices=False)

        numerical_rank = int((S > svd_eps).sum().item())
        full_rank = min(max(1, numerical_rank), U_svd.size(1))
        top_rank = min(max(1, int(top_rank)), full_rank)

        B_full = U_svd[:, :full_rank].contiguous()       # [D, full_rank]
        B_top = U_svd[:, :top_rank].contiguous()         # [D, top_rank]

        _, R_raw = self.compute_raw_residual()
        _, R_eff = self.compute_residual()

        def _energy_stats(R):
            R = R.float()
            total_energy = R.pow(2).sum().clamp_min(eps)

            # Since B is orthonormal:
            # ||R B B^T||_F^2 = ||R B||_F^2
            full_energy = (R @ B_full).pow(2).sum()
            top_energy = (R @ B_top).pow(2).sum()

            rho_full = full_energy / total_energy
            rho_top = top_energy / total_energy
            rho_full_minus_top = rho_full - rho_top
            rho_outside_full = 1.0 - rho_full

            return {
                "residual_norm": torch.sqrt(total_energy).item(),
                "rho_full_text": rho_full.item(),
                "rho_top_text": rho_top.item(),
                "rho_full_minus_top": rho_full_minus_top.item(),
                "rho_outside_full": rho_outside_full.item(),
            }

        return {
            "full_rank": full_rank,
            "top_rank": top_rank,
            "numerical_rank": numerical_rank,
            "singular_values_top10": S[: min(10, S.numel())].detach().cpu(),
            "raw": _energy_stats(R_raw),
            "effective": _energy_stats(R_eff),
        }

    def forward(self):
        T = self.base_text_features
        _, R = self.compute_residual()
        W = T + self.beta * R
        return W


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, base_text_features, B_T, B_Proj):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.prompt_learner = TextConditionedCompensator(
            cfg=cfg,
            base_text_features=base_text_features,
            B_T=B_T,
            B_Proj=B_Proj,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    def forward(self, image, return_text=False):
        image_features = self.encode_image_features(image)
        T, W = self.get_text_classifiers(normalize=True)

        logits = self.logit_scale.exp() * image_features @ W.t()

        if return_text:
            return logits, W, T

        return logits
    
    @torch.no_grad()
    def encode_image_features(self, image):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except Exception:
            image_features = self.image_encoder(image.float())

        image_features = F.normalize(image_features.float(), dim=-1)
        return image_features


    @torch.no_grad()
    def get_text_classifiers(self, normalize=True):
        """
        Return:
            T: base text classifier, [C, D]
            W: compensated classifier, [C, D]
        """
        T = self.prompt_learner.get_base_text_features().float()
        W = self.prompt_learner().float()

        if normalize:
            T = F.normalize(T, dim=-1)
            W = F.normalize(W, dim=-1)

        return T, W


@TRAINER_REGISTRY.register()
class OrthComp_adapter(TrainerX):
    """Minimal trainer for text-conditioned CLIP classifier compensation."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.OrthComp_adapter.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.OrthComp_adapter.PREC in ["fp32", "amp"]:
            clip_model.float()

        # print("Building minimal text-conditioned compensation model")
        # text_encoder = TextEncoder(clip_model)
        # base_text_features = get_base_text_features(cfg, classnames, clip_model, text_encoder)
        # print(f">> Base text features shape: {base_text_features.shape}")

        # basis_rank = int(cfg_get(cfg, "TRAINER.OrthComp_adapter.TEXT_BASIS_RANK", 128))
        # proj_rank = int(cfg_get(cfg, "TRAINER.OrthComp_adapter.TEXT_PROJ_RANK", 128))
        # basis_center = bool(cfg_get(cfg, "TRAINER.OrthComp_adapter.TEXT_BASIS_CENTER", False))
        # B_T, B_Proj, singular_values, text_rank = build_text_basis(
        #     base_text_features,
        #     rank=basis_rank,
        #     rank_proj=proj_rank,
        #     center=basis_center,
        # )
        print("Building minimal text-conditioned compensation model")
        text_encoder = TextEncoder(clip_model)

        # 当前 dataset 是目标数据集：OxfordPets / Caltech101 / Food101 / ...
        target_text_features = get_base_text_features(cfg, classnames, clip_model, text_encoder)
        print(f">> Target text features shape: {target_text_features.shape}")

        cross_dataset_eval = bool(cfg_get(cfg, "TRAINER.OrthComp_adapter.CROSS_DATASET_EVAL", False))

        if cross_dataset_eval:
            source_model_dir = cfg_get(cfg, "TRAINER.OrthComp_adapter.SOURCE_MODEL_DIR", "")
            source_load_epoch = int(cfg_get(cfg, "TRAINER.OrthComp_adapter.SOURCE_LOAD_EPOCH", -1))

            if not source_model_dir:
                raise ValueError(
                    "CROSS_DATASET_EVAL=True but SOURCE_MODEL_DIR is empty. "
                    "Please provide the ImageNet-trained checkpoint directory."
                )

            # 关键：使用 ImageNet source checkpoint 里的 B_T / B_Proj
            B_T, B_Proj = load_source_basis_from_checkpoint(
                model_dir=source_model_dir,
                epoch=source_load_epoch,
                model_name="model",
            )

            singular_values = None
            text_rank = B_T.shape[1]

            print(">> Cross-dataset mode:")
            print("   Use target dataset text features as T_tar")
            print("   Use ImageNet-trained source B_T/B_Proj")
            print(f"   T_tar shape: {target_text_features.shape}")
            print(f"   B_T_src shape: {B_T.shape}")
            print(f"   B_Proj_src shape: {B_Proj.shape}")

        else:
            basis_rank = int(cfg_get(cfg, "TRAINER.OrthComp_adapter.TEXT_BASIS_RANK", 128))
            proj_rank = int(cfg_get(cfg, "TRAINER.OrthComp_adapter.TEXT_PROJ_RANK", 128))
            basis_center = bool(cfg_get(cfg, "TRAINER.OrthComp_adapter.TEXT_BASIS_CENTER", False))

            B_T, B_Proj, singular_values, text_rank = build_text_basis(
                target_text_features,
                rank=basis_rank,
                rank_proj=proj_rank,
                center=basis_center,
            )

            print(f">> B_T shape: {B_T.shape}, selected rank: {text_rank}")
            print(
                f">> Top singular values: "
                f"{singular_values[: min(10, singular_values.numel())].detach().cpu().numpy()}"
            )

        self.model = CustomCLIP(
            cfg=cfg,
            clip_model=clip_model,
            base_text_features=target_text_features,
            B_T=B_T,
            B_Proj=B_Proj,
        )
        print(f">> B_T shape: {B_T.shape}, selected rank: {text_rank}")
        # print(f">> Top singular values: {singular_values[: min(10, singular_values.numel())].detach().cpu().numpy()}")

        # self.model = CustomCLIP(
        #     cfg=cfg,
        #     clip_model=clip_model,
        #     base_text_features=base_text_features,
        #     B_T=B_T,
        #     B_Proj=B_Proj
        # )

        print("Turning off gradients in CLIP encoders; keeping prompt_learner trainable")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(f"  trainable: {name}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()

        trainable_params = list(self.model.prompt_learner.parameters())
        self.optim = build_optimizer(self.model, cfg.OPTIM, trainable_params)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.OrthComp_adapter.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
        self.target_class_indices = None

        if cfg.DATASET.NAME == "ImageNetAFilter":
            from trainers.imagenet_a_r_indexes_v2 import find_imagenet_a_indexes
            self.target_class_indices = torch.tensor(
                find_imagenet_a_indexes(), dtype=torch.long
            )
            print(f">> Use ImageNet-A filtered evaluation with {len(self.target_class_indices)} classes")

        elif cfg.DATASET.NAME == "ImageNetRFilter":
            from trainers.imagenet_a_r_indexes_v2 import find_imagenet_r_indexes
            self.target_class_indices = torch.tensor(
                find_imagenet_r_indexes(), dtype=torch.long
            )
            print(f">> Use ImageNet-R filtered evaluation with {len(self.target_class_indices)} classes")

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.OrthComp_adapter.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label
    
    def model_inference(self, image):
        logits = self.model(image)

        if getattr(self, "target_class_indices", None) is not None:
            indices = self.target_class_indices.to(logits.device)
            logits = logits.index_select(dim=1, index=indices)

        return logits

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        cross_dataset_eval = bool(
            cfg_get(self.cfg, "TRAINER.OrthComp_adapter.CROSS_DATASET_EVAL", False)
        )

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_loaded = checkpoint["epoch"]

            # Ignore possible CoOp-style fixed token buffers if present.
            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)

            if cross_dataset_eval:
                # 关键：不要加载 ImageNet 的 1000-class text features
                remove_keys = [
                    k for k in list(state_dict.keys())
                    if k.endswith("prompt_learner.base_text_features")
                ]
                for k in remove_keys:
                    print(f">> Cross-dataset eval: skip loading {k}")
                    state_dict.pop(k)

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_loaded})')
            missing, unexpected = self._models[name].load_state_dict(state_dict, strict=False)

            print(f">> Missing keys: {missing}")
            print(f">> Unexpected keys: {unexpected}")
            
    @torch.no_grad()
    def report_residual_energy(self):
        """
        Print residual energy ratios in the full text subspace and top-k text subspace.
        """
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        prompt_learner = model.prompt_learner

        top_rank = int(cfg_get(self.cfg, "TRAINER.OrthComp_adapter.TEXT_PROJ_RANK", 32))
        center = bool(cfg_get(self.cfg, "TRAINER.OrthComp_adapter.TEXT_BASIS_CENTER", False))

        stats = prompt_learner.residual_energy_analysis(
            top_rank=top_rank,
            center=center,
        )

        print("\n========== Residual Energy Analysis ==========")
        print(f">> full text numerical rank: {stats['full_rank']}")
        print(f">> top text rank: {stats['top_rank']}")
        print(f">> top singular values: {stats['singular_values_top10'].numpy()}")

        for mode in ["raw", "effective"]:
            s = stats[mode]
            print(f"\n[{mode.upper()} residual]")
            print(f"  ||R||_F                 : {s['residual_norm']:.6f}")
            print(f"  rho_full_text           : {100.0 * s['rho_full_text']:.4f}%")
            print(f"  rho_top{stats['top_rank']}_text        : {100.0 * s['rho_top_text']:.4f}%")
            print(f"  rho_full_minus_top      : {100.0 * s['rho_full_minus_top']:.4f}%")
            print(f"  rho_outside_full_text   : {100.0 * s['rho_outside_full']:.4f}%")

        print("==============================================\n")

    @torch.no_grad()
    def _collect_image_features_for_gap(self, data_loader):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.eval()

        all_features = []
        all_labels = []

        for batch in tqdm(data_loader, desc="Extract image features"):
            image = batch["img"].to(self.device)
            label = batch["label"].to(self.device)

            feat = model.encode_image_features(image)

            all_features.append(feat.detach().cpu())
            all_labels.append(label.detach().cpu())

        Z = torch.cat(all_features, dim=0)      # [N, D]
        y = torch.cat(all_labels, dim=0).long() # [N]

        print(f">> Collected image features: {Z.shape}")
        print(f">> Collected labels: {y.shape}")

        return Z, y

    @torch.no_grad()
    def _compute_pair_metrics(self, Z, y, Q, chunk_size=4096):
        """
        Args:
            Z: normalized image features, [N, D], CPU tensor
            y: labels, [N], CPU tensor
            Q: normalized classifier, [C, D], CPU tensor

        Return:
            dict with acc, pos, neg, sep, margin
        """
        device = self.device

        Z = F.normalize(Z.float(), dim=-1)
        Q = F.normalize(Q.float(), dim=-1)

        C = Q.size(0)
        N = Z.size(0)

        q = Q.to(device)

        total_correct = 0
        total_pos = 0.0
        total_neg = 0.0
        total_margin = 0.0
        total_num = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            z = Z[start:end].to(device)
            yy = y[start:end].to(device)

            sims = z @ q.t()  # [B, C]

            bsz = sims.size(0)
            row_idx = torch.arange(bsz, device=device)

            pos = sims[row_idx, yy]

            neg = (sims.sum(dim=1) - pos) / max(C - 1, 1)

            sims_neg = sims.clone()
            sims_neg[row_idx, yy] = -1e9
            max_neg = sims_neg.max(dim=1).values
            margin = pos - max_neg

            pred = sims.argmax(dim=1)
            correct = pred.eq(yy).sum().item()

            total_correct += correct
            total_pos += pos.sum().item()
            total_neg += neg.sum().item()
            total_margin += margin.sum().item()
            total_num += bsz

        pos_mean = total_pos / total_num
        neg_mean = total_neg / total_num

        return {
            "acc": 100.0 * total_correct / total_num,
            "pos": pos_mean,
            "neg": neg_mean,
            "sep": pos_mean - neg_mean,
            "margin": total_margin / total_num,
        }
        
    @torch.no_grad()
    def _compute_prototypes(self, Z, y, num_classes):
        """
        Args:
            Z: normalized image features, [N, D]
            y: labels, [N]
        Return:
            P: normalized visual prototypes, [C, D]
            valid: whether class appears in test set, [C]
        """
        Z = F.normalize(Z.float(), dim=-1)

        N, D = Z.shape
        sums = torch.zeros(num_classes, D, dtype=torch.float32)
        counts = torch.zeros(num_classes, dtype=torch.float32)

        sums.index_add_(0, y, Z)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

        valid = counts > 0
        P = sums / counts.clamp_min(1.0).unsqueeze(1)
        P = F.normalize(P, dim=-1)

        return P, valid

    @torch.no_grad()
    def _compute_prototypes(self, Z, y, num_classes):
        """
        Args:
            Z: normalized image features, [N, D]
            y: labels, [N]
        Return:
            P: normalized visual prototypes, [C, D]
            valid: whether class appears in test set, [C]
        """
        Z = F.normalize(Z.float(), dim=-1)

        N, D = Z.shape
        sums = torch.zeros(num_classes, D, dtype=torch.float32)
        counts = torch.zeros(num_classes, dtype=torch.float32)

        sums.index_add_(0, y, Z)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

        valid = counts > 0
        P = sums / counts.clamp_min(1.0).unsqueeze(1)
        P = F.normalize(P, dim=-1)

        return P, valid


    @torch.no_grad()
    def _compute_proto_gap(self, Z, y, Q):
        Q = F.normalize(Q.float(), dim=-1)
        num_classes = Q.size(0)

        P, valid = self._compute_prototypes(Z, y, num_classes)

        align = (P[valid] * Q[valid]).sum(dim=-1)
        proto_align = align.mean().item()
        proto_gap = 1.0 - proto_align

        return {
            "proto_align": proto_align,
            "proto_gap": proto_gap,
            "num_valid_classes": int(valid.sum().item()),
        }, P, valid
        
    @torch.no_grad()
    def _spectral_basis_from_rows(self, X, energy=0.95, topk=None, center=False):
        """
        Args:
            X: [N, D] or [C, D], rows are samples/classifiers.
        Return:
            B: [D, r], orthonormal basis in feature dimension.
            info: rank and singular values.
        """
        X = X.float()

        if center:
            X = X - X.mean(dim=0, keepdim=True)

        # SVD on X^T so that U lives in feature dimension.
        # For ImageNet-1K, X can be [50000, 512], so X.T is [512, 50000].
        svd_device = self.device if torch.cuda.is_available() else torch.device("cpu")
        Xt = X.t().contiguous().to(svd_device)

        U, S, _ = torch.linalg.svd(Xt, full_matrices=False)

        if topk is not None and int(topk) > 0:
            r = min(int(topk), U.size(1))
        else:
            e = S.pow(2)
            cum = torch.cumsum(e, dim=0) / e.sum().clamp_min(1e-12)
            r = int(torch.searchsorted(cum, torch.tensor(float(energy), device=cum.device)).item()) + 1
            r = min(max(1, r), U.size(1))

        B = U[:, :r].contiguous().cpu()

        info = {
            "rank": r,
            "top_singular_values": S[:10].detach().cpu().tolist(),
            "energy_ratio": float(
                S[:r].pow(2).sum().detach().cpu()
                / S.pow(2).sum().detach().cpu().clamp_min(1e-12)
            ),
        }

        return B, info


    @torch.no_grad()
    def _directional_basis_gap(self, B_img, B_cls):
        """
        d(B_i, B_q) = mean_j || b_i_j - B_q B_q^T b_i_j ||_2

        Args:
            B_img: [D, r_i]
            B_cls: [D, r_q]
        """
        B_img = B_img.float()
        B_cls = B_cls.float()

        proj = B_cls @ (B_cls.t() @ B_img)
        residual = B_img - proj

        return residual.norm(dim=0).mean().item()
    
    @torch.no_grad()
    def _normalize_rows_safe(self, X, eps=1e-12):
        """
        Row-normalize a matrix before building a subspace.
        This removes scale effects and keeps only direction information.
        """
        X = X.float()
        return X / X.norm(dim=-1, keepdim=True).clamp_min(eps)


    def _basis_gap_tag(self, energy=0.95, topk=None):
        """
        Build a clean name tag for JSON keys.
        Examples:
            topk=32      -> top32
            topk=None,
            energy=0.95 -> energy95
        """
        if topk is not None and int(topk) > 0:
            return f"top{int(topk)}"
        return f"energy{int(round(float(energy) * 100))}"


    @torch.no_grad()
    def _basis_gap_to_rows(
        self,
        Z,
        A,
        energy=0.95,
        topk=None,
        center=False,
    ):
        """
        Compute d(B_Z, B_A), where rows of A define the target direction space.

        Args:
            Z: [N, D], image features
            A: [M, D], classifier / induced direction rows
            energy: energy ratio for adaptive-rank basis, e.g. 0.95
            topk:
                - if topk is not None and > 0, use fixed top-k basis
                - if topk is None or <= 0, use energy-preserving basis
            center: whether to center rows before SVD
        """
        Z = self._normalize_rows_safe(Z)
        A = self._normalize_rows_safe(A)

        if topk is not None and int(topk) <= 0:
            topk = None

        B_Z, info_Z = self._spectral_basis_from_rows(
            Z,
            energy=energy,
            topk=topk,
            center=center,
        )
        B_A, info_A = self._spectral_basis_from_rows(
            A,
            energy=energy,
            topk=topk,
            center=center,
        )

        gap = self._directional_basis_gap(B_Z, B_A)

        return gap, {
            "image_basis": info_Z,
            "target_basis": info_A,
        }


    @torch.no_grad()
    def analyze_modality_gap(
        self,
        save_path=None,
        energy=0.95,
        topk=None,
        center=False,
    ):
        """
        Minimal feature-space modality gap analysis for Ours.

        We report:

            CLIP gap:
                d(B_Z, B_T)

            Realized gap:
                d(B_Z, B_W)

            Induced gap:
                d(B_Z, B_[T;R]), where R = W - T

        Basis choice:
            - topk=None: use 95% energy basis by default
            - topk=32  : use fixed top-32 basis
        """
        print("\n========== Minimal Modality Gap Analysis: Ours ==========")

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.eval()

        # -------------------------------------------------------
        # 1. Extract image features
        # -------------------------------------------------------
        Z, y = self._collect_image_features_for_gap(self.test_loader)
        Z = self._normalize_rows_safe(Z)
        y = y.long()

        # -------------------------------------------------------
        # 2. Get raw T and W
        # -------------------------------------------------------
        # Use normalize=False so that R = W - T reflects the actual learned residual.
        # We will row-normalize each direction before SVD anyway.
        T_raw, W_raw = model.get_text_classifiers(normalize=False)

        T_raw = T_raw.detach().cpu().float()  # [C, D]
        W_raw = W_raw.detach().cpu().float()  # [C, D]

        # Effective residual direction.
        # Since W = T + beta R, W - T already includes beta.
        # Scale does not affect the subspace after row normalization.
        R_raw = W_raw - T_raw                 # [C, D]

        # Remove near-zero residual rows for numerical stability.
        r_norm = R_raw.norm(dim=-1)
        valid_r = r_norm > 1e-12
        R_valid = R_raw[valid_r]

        if R_valid.size(0) == 0:
            raise RuntimeError(
                "All residual rows are nearly zero. "
                "Please check whether the compensator was loaded/trained correctly."
            )

        # Induced direction space: [T; R]
        TR_rows = torch.cat([T_raw, R_valid], dim=0)

        if topk is not None and int(topk) <= 0:
            topk = None

        tag = self._basis_gap_tag(energy=energy, topk=topk)

        print(f">> Dataset      : {self.cfg.DATASET.NAME}")
        print(f">> Z shape      : {tuple(Z.shape)}")
        print(f">> T shape      : {tuple(T_raw.shape)}")
        print(f">> W shape      : {tuple(W_raw.shape)}")
        print(f">> R valid rows : {int(valid_r.sum().item())}/{R_raw.size(0)}")
        print(f">> basis tag    : {tag}")
        print(f">> energy       : {energy}")
        print(f">> topk         : {topk}")
        print(f">> center       : {center}")

        # -------------------------------------------------------
        # 3. Compute basis gaps
        # -------------------------------------------------------
        gap_T, info_T = self._basis_gap_to_rows(
            Z=Z,
            A=T_raw,
            energy=energy,
            topk=topk,
            center=center,
        )

        gap_W, info_W = self._basis_gap_to_rows(
            Z=Z,
            A=W_raw,
            energy=energy,
            topk=topk,
            center=center,
        )

        gap_TR, info_TR = self._basis_gap_to_rows(
            Z=Z,
            A=TR_rows,
            energy=energy,
            topk=topk,
            center=center,
        )

        # -------------------------------------------------------
        # 4. Accuracy sanity check only
        # -------------------------------------------------------
        T = self._normalize_rows_safe(T_raw)
        W = self._normalize_rows_safe(W_raw)

        logits_T = Z @ T.t()
        logits_W = Z @ W.t()

        acc_T = 100.0 * logits_T.argmax(dim=1).eq(y).float().mean().item()
        acc_W = 100.0 * logits_W.argmax(dim=1).eq(y).float().mean().item()

        # -------------------------------------------------------
        # 5. Summarize
        # -------------------------------------------------------
        results = {
            "method": "Ours",
            "analysis_type": "feature_space_basis_gap",
            "dataset": self.cfg.DATASET.NAME,
            "num_samples": int(Z.size(0)),
            "num_classes": int(T_raw.size(0)),
            "feature_dim": int(T_raw.size(1)),
            "energy": float(energy),
            "topk": None if topk is None else int(topk),
            "basis_tag": tag,
            "center": bool(center),

            "accuracy_sanity_check": {
                "clip_T_acc": acc_T,
                "ours_W_acc": acc_W,
                "acc_gain": acc_W - acc_T,
            },

            "basis_gap": {
                f"clip_T_gap_{tag}": gap_T,
                f"realized_W_gap_{tag}": gap_W,
                f"induced_T_plus_R_gap_{tag}": gap_TR,

                f"realized_gap_reduction_{tag}": gap_T - gap_W,
                f"induced_gap_reduction_{tag}": gap_T - gap_TR,
            },

            "residual_info": {
                "num_residual_rows": int(R_raw.size(0)),
                "num_valid_residual_rows": int(valid_r.sum().item()),
                "mean_residual_norm": float(r_norm.mean().item()),
                "max_residual_norm": float(r_norm.max().item()),
                "min_residual_norm": float(r_norm.min().item()),
            },

            "basis_info": {
                "clip_T": info_T,
                "realized_W": info_W,
                "induced_T_plus_R": info_TR,
            },
        }

        print("\n========== Feature-space Basis Gap ==========")
        print(f"Basis tag              : {tag}")
        print(f"CLIP T gap             : {gap_T:.6f}")
        print(f"Realized W gap          : {gap_W:.6f}")
        print(f"Induced [T; R] gap      : {gap_TR:.6f}")
        print(f"Realized reduction      : {gap_T - gap_W:.6f}")
        print(f"Induced reduction       : {gap_T - gap_TR:.6f}")

        print("\n========== Basis Rank Info ==========")
        print(f"Image basis rank        : {info_T['image_basis']['rank']}")
        print(f"T basis rank            : {info_T['target_basis']['rank']}")
        print(f"W basis rank            : {info_W['target_basis']['rank']}")
        print(f"[T; R] basis rank       : {info_TR['target_basis']['rank']}")

        print("\n========== Accuracy Sanity Check ==========")
        print(f"CLIP T acc              : {acc_T:.4f}")
        print(f"Ours W acc              : {acc_W:.4f}")
        print(f"Acc gain                : {acc_W - acc_T:.4f}")
        print("============================================\n")

        if save_path is None or save_path == "":
            save_path = osp.join(self.cfg.OUTPUT_DIR, f"modality_gap_basis_ours_{tag}.json")

        os.makedirs(osp.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f">> Saved minimal modality gap analysis to: {save_path}")

        return results