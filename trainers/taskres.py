'''
Task Residual Tuning
by Tao Yu (yutao666@mail.ustc.edu.cn)
Oct 4, 2022
'''
import os
import os.path as osp
from re import template

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from trainers.utils_features import extract_image_features, save_feature_cache, load_feature_cache
from trainers.utils_probe import (
    fit_surrogate_probe,
    save_probe_cache,
    load_probe_cache,
)
from trainers.utils_subspace import (
    build_text_basis,
    compute_probe_residual,
    build_residual_basis,
    save_subspace_cache,
    load_subspace_cache,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

_tokenizer = _Tokenizer()

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

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def _get_feature_cache_path(cfg):
    backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
    dataset = cfg.DATASET.NAME
    shots = cfg.DATASET.NUM_SHOTS
    seed = cfg.SEED
    cache_dir = cfg.TRAINER.TaskRes.FEAT_CACHE_DIR
    filename = f"{dataset}_{backbone}_{shots}shots_seed{seed}.pt"
    return osp.join(cache_dir, filename)

def _get_probe_cache_path(cfg):
    backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
    dataset = cfg.DATASET.NAME
    shots = cfg.DATASET.NUM_SHOTS
    seed = cfg.SEED
    probe_dir = cfg.TRAINER.TaskRes.PROBE_DIR
    filename = f"{dataset}_{backbone}_{shots}shots_seed{seed}_probe.pt"
    return osp.join(probe_dir, filename)

def _get_subspace_cache_path(cfg):
    backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
    dataset = cfg.DATASET.NAME
    shots = cfg.DATASET.NUM_SHOTS
    seed = cfg.SEED
    k = cfg.TRAINER.TaskRes.RESIDUAL_RANK
    subspace_dir = cfg.TRAINER.TaskRes.SUBSPACE_DIR
    filename = f"{dataset}_{backbone}_{shots}shots_seed{seed}_k{k}_subspace.pt"
    return osp.join(subspace_dir, filename)

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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

# TaskRes(-Text)
class TaskResLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, base_text_features):
        super().__init__()
        self.device = clip_model.dtype
        self.alpha = cfg.TRAINER.TaskRes.RESIDUAL_SCALE
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features)
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(self):
        return self.base_text_features + self.alpha * self.text_feature_residuals   # t + a * x

# # TaskRes-Image
# class TaskResLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model, base_text_features):
#         super().__init__()
#         self.device = clip_model.dtype
#         # feat_dim = base_text_features.size(-1)
#         self.alpha = cfg.TRAINER.TaskRes.RESIDUAL_SCALE
#         print(">> DCT scale factor: ", self.alpha)
#         self.register_buffer("base_text_features", base_text_features)
#         self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features[0:1]))

#     def forward(self):
#         # print(self.base_text_features.dtype, self.text_feature_residuals.dtype)
#         return self.base_text_features, self.alpha * self.text_feature_residuals

def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)

def _get_enhanced_base_text_features(cfg, classnames, clip_model, text_encoder, pretraiend_model):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

        pretrained_text_projection = torch.load(pretraiend_model)

        state_dict = text_encoder.state_dict()
        state_dict['text_projection'] = pretrained_text_projection['state_dict']['weight'].t()
        text_encoder.load_state_dict(state_dict)
        print(">> Pretrained text encoder loaded!")
        params = pretrained_text_projection['state_dict']['weight'].size(0) * \
            pretrained_text_projection['state_dict']['weight'].size(1)
        print(">> Text projection parameters: ", params)
        print(pretrained_text_projection['state_dict'].keys())
    
    dataset = cfg.DATASET.NAME
    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)

# TaskRes by Tao Yu, Oct 4, 2022
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype   # float16
        text_encoder = TextEncoder(clip_model)
        if cfg.TRAINER.TaskRes.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        else:
            print(">> Use enhanced base!")
            base_text_features = _get_enhanced_base_text_features(
                cfg, classnames, clip_model, text_encoder, cfg.TRAINER.TaskRes.ENHANCED_BASE)

        self.prompt_learner = TaskResLearner(cfg, classnames, clip_model, base_text_features)

    def forward(self, image):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except:
            image_features = self.image_encoder(image.float())

        # TaskRes-Text
        text_features = self.prompt_learner()

        # # TaskRes-Image
        # text_features, image_res = self.prompt_learner()
        # image_features += image_res

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

@TRAINER_REGISTRY.register()
class TaskRes(TrainerX):
    """Context Optimization (TaskRes).

    Task Residual for Tuning Vision-Language Models
    https://arxiv.org/abs/2211.10277
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TaskRes.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.TaskRes.PREC == "fp32" or cfg.TRAINER.TaskRes.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")

        text_encoder = TextEncoder(clip_model)

        if cfg.TRAINER.TaskRes.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        else:
            print(">> Use enhanced base!")
            base_text_features = _get_enhanced_base_text_features(
                cfg, classnames, clip_model, text_encoder, cfg.TRAINER.TaskRes.ENHANCED_BASE
            )


        print(f">> Base text features shape: {base_text_features.shape}")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # # Step 1: build frozen image feature cache for later probe fitting
        # if cfg.TRAINER.TaskRes.BUILD_FEAT_CACHE:
        #     cache_path = _get_feature_cache_path(cfg)

        #     if osp.exists(cache_path) and not cfg.TRAINER.TaskRes.FEAT_CACHE_FORCE_REBUILD:
        #         print(f">> Found existing feature cache: {cache_path}")
        #         feats, labels = load_feature_cache(cache_path)
        #     else:
        #         print(">> Building training feature cache from frozen CLIP image encoder...")
        #         image_encoder = clip_model.visual.to(self.device)
        #         image_encoder.eval()

        #         feats, labels = extract_image_features(
        #             data_loader=self.train_loader_x,
        #             image_encoder=image_encoder,
        #             device=self.device,
        #             dtype=clip_model.dtype,
        #             normalize=True
        #         )
        #         save_feature_cache(cache_path, feats, labels)
        #         print(f">> Feature cache saved to: {cache_path}")

        #         print("len(train_loader_x):", len(self.train_loader_x))              # batch 数
        #         print("len(train_loader_x.dataset):", len(self.train_loader_x.dataset))  # 数据集样本数
        #         print("extracted feature num:", feats.shape[0])                      # 实际提取样本数

        #     print(f">> Cached train features shape: {feats.shape}")
        #     print(f">> Cached train labels shape: {labels.shape}")

        # # Step 2: fit/load surrogate probe on frozen image features
        # if cfg.TRAINER.TaskRes.BUILD_PROBE:
        #     probe_path = _get_probe_cache_path(cfg)

        #     if osp.exists(probe_path) and not cfg.TRAINER.TaskRes.PROBE_FORCE_REBUILD:
        #         print(f">> Found existing probe cache: {probe_path}")
        #         probe_weight, probe_acc, probe_meta = load_probe_cache(probe_path)
        #     else:
        #         print(">> Fitting surrogate probe on frozen CLIP image features...")
        #         probe_weight, probe_acc = fit_surrogate_probe(
        #             train_feats=feats,
        #             train_labels=labels,
        #             num_classes=self.num_classes,
        #             lr=cfg.TRAINER.TaskRes.PROBE_LR,
        #             weight_decay=cfg.TRAINER.TaskRes.PROBE_WD,
        #             epochs=cfg.TRAINER.TaskRes.PROBE_EPOCHS,
        #             batch_size=cfg.TRAINER.TaskRes.PROBE_BATCH_SIZE,
        #             device=self.device,
        #             bias=False,
        #         )

        #         probe_meta = {
        #             "dataset": cfg.DATASET.NAME,
        #             "backbone": cfg.MODEL.BACKBONE.NAME,
        #             "num_shots": cfg.DATASET.NUM_SHOTS,
        #             "seed": cfg.SEED,
        #             "num_classes": self.num_classes,
        #             "feat_dim": feats.size(1),
        #         }
        #         save_probe_cache(probe_path, probe_weight, probe_acc, probe_meta)
        #         print(f">> Probe cache saved to: {probe_path}")

        #     print(f">> Probe weight shape: {probe_weight.shape}")
        #     if probe_acc is not None:
        #         print(f">> Probe cached train acc: {probe_acc:.2f}%")

        #     # keep for later subspace construction
        #     self.probe_weight = probe_weight

        # # Step 3: build/load text basis and residual basis
        # if cfg.TRAINER.TaskRes.BUILD_SUBSPACE:
        #     subspace_path = _get_subspace_cache_path(cfg)

        #     if osp.exists(subspace_path) and not cfg.TRAINER.TaskRes.SUBSPACE_FORCE_REBUILD:
        #         print(f">> Found existing subspace cache: {subspace_path}")
        #         subspace_data = load_subspace_cache(subspace_path)
        #         B_T = subspace_data["B_T"]
        #         B_R = subspace_data["B_R"]
        #         text_rank = subspace_data["text_rank"]
        #         residual_singular_values = subspace_data["residual_singular_values"]
        #     else:
        #         print(">> Building text subspace basis B_T ...")
        #         B_T, text_singular_values, text_rank = build_text_basis(base_text_features)

        #         print(">> Computing probe residual outside text subspace ...")
        #         W_res = compute_probe_residual(self.probe_weight, B_T)

        #         print(">> Building low-rank residual basis B_R ...")
        #         B_R, residual_singular_values = build_residual_basis(
        #             W_res,
        #             k=cfg.TRAINER.TaskRes.RESIDUAL_RANK
        #         )

        #         save_subspace_cache(
        #             subspace_path,
        #             B_T=B_T,
        #             B_R=B_R,
        #             text_rank=text_rank,
        #             residual_singular_values=residual_singular_values,
        #             meta={
        #                 "dataset": cfg.DATASET.NAME,
        #                 "backbone": cfg.MODEL.BACKBONE.NAME,
        #                 "num_shots": cfg.DATASET.NUM_SHOTS,
        #                 "seed": cfg.SEED,
        #                 "text_feat_shape": list(base_text_features.shape),
        #                 "probe_weight_shape": list(self.probe_weight.shape),
        #             }
        #         )
        #         print(f">> Subspace cache saved to: {subspace_path}")

        #     print(f">> B_T shape: {B_T.shape}, text rank: {text_rank}")
        #     print(f">> B_R shape: {B_R.shape}")
        #     self.B_T = B_T
        #     self.B_R = B_R
        #     # orth_check = torch.norm(B_T.t() @ B_R).item()
        #     # print(f">> ||B_T^T B_R||_F = {orth_check:.6e}")
        #     # print(f">> Residual singular values (top 10): {residual_singular_values[:10]}")

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.TaskRes.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.TaskRes.PREC
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
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            if self.cfg.DATASET.NAME == 'ImageNetA' or self.cfg.DATASET.NAME == 'ImageNetR':
                if self.cfg.DATASET.NAME == 'ImageNetA':
                    from .imagenet_a_r_indexes_v2 import find_imagenet_a_indexes as find_indexes
                else:
                    from .imagenet_a_r_indexes_v2 import find_imagenet_r_indexes as find_indexes
                imageneta_indexes = find_indexes()
                state_dict['base_text_features'] = state_dict['base_text_features'][imageneta_indexes]
                state_dict['text_feature_residuals'] = state_dict['text_feature_residuals'][imageneta_indexes]

            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
