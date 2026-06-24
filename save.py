# '''
# Task Residual Tuning
# by Tao Yu (yutao666@mail.ustc.edu.cn)
# Oct 4, 2022
# '''
# import os
# import os.path as osp
# from re import template

# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from torch.cuda.amp import GradScaler, autocast

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.metrics import compute_accuracy
# from dassl.utils import load_pretrained_weights, load_checkpoint
# from dassl.optim import build_optimizer, build_lr_scheduler

# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from trainers.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
# from trainers.utils_features import extract_image_features, save_feature_cache, load_feature_cache
# from trainers.utils_probe import (
#     fit_surrogate_probe,
#     save_probe_cache,
#     load_probe_cache,
#     solve_ridge_init_G
# )
# from trainers.utils_subspace import (
#     build_text_basis,
#     build_reference_text_basis,
#     compute_probe_residual,
#     build_residual_basis,
#     save_subspace_cache,
#     load_subspace_cache,
# )


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.allow_tf32 = True

# _tokenizer = _Tokenizer()

# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a photo of a {}, a type of pet.",
#     "OxfordFlowers": "a photo of a {}, a type of flower.",
#     "FGVCAircraft": "a photo of a {}, a type of aircraft.",
#     "DescribableTextures": "{} texture.",
#     "EuroSAT": "a centered satellite photo of {}.",
#     "StanfordCars": "a photo of a {}.",
#     "Food101": "a photo of {}, a type of food.",
#     "SUN397": "a photo of a {}.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a person doing {}.",
#     "ImageNet": "a photo of a {}.",
#     "ImageNetSketch": "a photo of a {}.",
#     "ImageNetV2": "a photo of a {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",
# }

# CUSTOM_TEMPLATE_ENSEMBLES = {
#     "OxfordFlowers": [
#         "a photo of a {}, a type of flower.",
#         "a close-up photo of a {} flower.",
#         "a photo of the flower {}.",
#     ],

#     "Food101": [
#         "a photo of {}, a type of food.",
#         "a close-up photo of {}.",
#         "a photo of a plate of {}.",
#         "a photo of freshly made {}.",
#         "a restaurant photo of {}.",
#     ],

#     "OxfordPets": [
#         "a photo of a {}, a type of pet.",
#         "a photo of the pet {}.",
#         "a close-up photo of a {}.",
#         "a photo of a {} animal.",
#     ],

#     "FGVCAircraft": [
#         "a photo of a {}, a type of aircraft.",
#         "a photo of the aircraft {}.",
#         "a side view photo of a {}.",
#         "a photo of a {} airplane.",
#     ],

#     "StanfordCars": [
#         "a photo of a {}.",
#         "a photo of the car {}.",
#         "a side view photo of a {}.",
#         "a photo of a {} vehicle.",
#     ],
# }

# def _get_prompt_templates(cfg):
#     dataset = cfg.DATASET.NAME

#     try:
#         use_prompt_ensemble = cfg.TRAINER.OrthComp_adapter.PROMPT_ENSEMBLE
#     except Exception:
#         use_prompt_ensemble = False

#     if dataset == "ImageNet":
#         return IMAGENET_TEMPLATES_SELECT

#     if use_prompt_ensemble and dataset in CUSTOM_TEMPLATE_ENSEMBLES:
#         return CUSTOM_TEMPLATE_ENSEMBLES[dataset]

#     return [CUSTOM_TEMPLATES[dataset]]

# def load_clip_to_cpu(cfg):
#     backbone_name = cfg.MODEL.BACKBONE.NAME
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")

#     model = clip.build_model(state_dict or model.state_dict())

#     return model

# def _get_feature_cache_path(cfg):
#     backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
#     dataset = cfg.DATASET.NAME
#     shots = cfg.DATASET.NUM_SHOTS
#     seed = cfg.SEED
#     cache_dir = cfg.TRAINER.OrthComp_adapter.FEAT_CACHE_DIR
#     filename = f"{dataset}_{backbone}_{shots}shots_seed{seed}.pt"
#     return osp.join(cache_dir, filename)

# def _get_probe_cache_path(cfg):
#     backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
#     dataset = cfg.DATASET.NAME
#     shots = cfg.DATASET.NUM_SHOTS
#     seed = cfg.SEED
#     probe_dir = cfg.TRAINER.OrthComp_adapter.PROBE_DIR
#     filename = f"{dataset}_{backbone}_{shots}shots_seed{seed}_probe.pt"
#     return osp.join(probe_dir, filename)

# def _get_subspace_cache_path(cfg):
#     backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
#     dataset = cfg.DATASET.NAME
#     shots = cfg.DATASET.NUM_SHOTS
#     seed = cfg.SEED
#     k = cfg.TRAINER.OrthComp_adapter.RESIDUAL_RANK
#     subspace_dir = cfg.TRAINER.OrthComp_adapter.SUBSPACE_DIR
#     filename = f"{dataset}_{backbone}_{shots}shots_seed{seed}_k{k}_subspace.pt"
#     return osp.join(subspace_dir, filename)

# def _get_ref_text_cache_path(cfg):
#     backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
#     # ref_name = osp.basename(cfg.TRAINER.OrthComp_adapter.REF_LABEL_FILE).replace(".txt", "")
#     cache_dir = cfg.TRAINER.OrthComp_adapter.REF_TEXT_CACHE_DIR
#     filename = f"ImageNet21K_{backbone}_text_features.pt"
#     return osp.join(cache_dir, filename)

# def classwise_tangent_projection(R, T):
#     """
#     R: [C, D], generated residual
#     T: [C, D], original text features
#     """
#     T_hat = F.normalize(T, dim=-1)
#     R_tan = R - (R * T_hat).sum(dim=-1, keepdim=True) * T_hat
#     return R_tan

# def build_class_prototypes_from_features(feats, labels, num_classes, normalize=True):
#     """
#     Build class-wise visual prototypes from frozen CLIP image features.

#     Args:
#         feats:  [N, D], usually normalized image features
#         labels: [N]
#         num_classes: number of classes
#         normalize: whether to normalize prototypes

#     Returns:
#         prototypes: [C, D]
#         counts:     [C]
#     """
#     feats = feats.float()
#     labels = labels.long()

#     device = feats.device
#     D = feats.size(1)

#     prototypes = torch.zeros(num_classes, D, device=device, dtype=feats.dtype)
#     counts = torch.zeros(num_classes, device=device, dtype=feats.dtype)

#     prototypes.index_add_(0, labels, feats)
#     counts.index_add_(0, labels, torch.ones_like(labels, dtype=feats.dtype))

#     prototypes = prototypes / counts.clamp_min(1.0).unsqueeze(1)

#     if normalize:
#         prototypes = F.normalize(prototypes, dim=-1)

#     return prototypes, counts










# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype

#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

#         return x

# # TaskRes(-Text)
# class TaskResLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model, base_text_features):
#         super().__init__()
#         self.device = clip_model.dtype
#         self.alpha = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE
#         print(">> DCT scale factor: ", self.alpha)
#         self.register_buffer("base_text_features", base_text_features)
#         self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

#     def forward(self):
#         return self.base_text_features + self.alpha * self.text_feature_residuals   # t + a * x
    
# class OrthCompLearner(nn.Module):
#     def __init__(self, cfg, base_text_features, B_T, B_R, G_init=None):
#         super().__init__()

#         self.beta = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE

#         # buffers: fixed geometry objects
#         self.register_buffer("base_text_features", base_text_features.float())  # [K, D]
#         self.register_buffer("B_T", B_T.float())                                # [D, r]
#         self.register_buffer("B_R", B_R.float())                                # [D, k]

#         r = B_T.shape[1]
#         k = B_R.shape[1]

#         # the only learnable parameter
#         self.G = nn.Parameter(torch.zeros(r, k))
#         self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

#     def forward(self):
#         T = self.base_text_features          # [K, D]
#         U = T @ self.B_T                     # [K, r]
#         A = U @ self.G                       # [K, k]
#         R = A @ self.B_R.t()                 # [K, D]
#         W = T + self.beta * R                # [K, D]
#         return W
    
# class OrthCompProjLearner(nn.Module):
#     def __init__(self, cfg, base_text_features, B_T, B_R, G_init=None):
#         super().__init__()

#         self.beta = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE
#         self.register_buffer("base_text_features", base_text_features.float())
#         self.register_buffer("B_T", B_T.float())

#         K, D = base_text_features.shape
#         r = B_T.shape[1]

#         if G_init is None:
#             G_init = torch.zeros(r, D, dtype=base_text_features.dtype, device=base_text_features.device)
#         else:
#             G_init = G_init.float()

#         self.G = nn.Parameter(G_init.clone())

#         self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

#     def forward(self):
#         T = self.base_text_features
#         U = T @ self.B_T
#         S = U @ self.G
#         R = S - S @ self.B_T @ self.B_T.t()
#         W = T + self.beta * R 
#         return W
    

# class LowRankClassSpecificLearner(nn.Module):
#     def __init__(self, cfg, base_text_features, B_T, B_R):
#         super().__init__()

#         self.beta = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE

#         # fixed parts
#         self.register_buffer("base_text_features", base_text_features.float())  # [K, D]
#         self.register_buffer("B_R", B_R.float())                                # [D, k]

#         K = base_text_features.shape[0]
#         k = B_R.shape[1]

#         # learnable class-specific low-rank coefficients
#         self.A = nn.Parameter(torch.zeros(K, k))                                # [K, k]

#     def forward(self):
#         T = self.base_text_features   # [K, D]
#         R = self.A @ self.B_R.t()     # [K, D]
#         W = T + self.beta * R         # [K, D]
#         return W

# class OrthCompHybridLearner(nn.Module):
#     def __init__(self, cfg, base_text_features, B_T, B_R):
#         super().__init__()

#         self.beta = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE

#         self.register_buffer("base_text_features", base_text_features.float())  # [K, D]
#         self.register_buffer("B_T", B_T.float())                                # [D, r]
#         self.register_buffer("B_R", B_R.float())                                # [D, k]

#         K = base_text_features.shape[0]
#         r = B_T.shape[1]
#         k = B_R.shape[1]

#         # shared generator
#         self.G = nn.Parameter(torch.zeros(r, k))

#         # class-specific correction
#         self.C = nn.Parameter(torch.zeros(K, k))

#     def forward(self):
#         T = self.base_text_features              # [K, D]
#         U = T @ self.B_T                         # [K, r]
#         A_shared = U @ self.G                    # [K, k]
#         A = A_shared + self.C                    # [K, k]
#         R = A @ self.B_R.t()                     # [K, D]
#         W = T + self.beta * R                    # [K, D]
#         return W
    
# class OrthCompProjGateLearner(nn.Module):
#     def __init__(self, cfg, base_text_features, B_T, B_R):
#         super().__init__()

#         self.beta = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE

#         self.register_buffer("base_text_features", base_text_features.float())  # [K, D]
#         self.register_buffer("B_T", B_T.float())                                # [D, r]

#         K, D = base_text_features.shape
#         r = B_T.shape[1]

#         # shared generator
#         self.G = nn.Parameter(torch.zeros(r, D))   # [r, D]

#         # text-conditioned scalar gate parameter
#         self.q = nn.Parameter(torch.zeros(r, 1))   # [r, 1]

#     def compute_residual(self):
#         T = self.base_text_features              # [K, D]
#         U = T @ self.B_T                         # [K, r]
#         S = U @ self.G                           # [K, D]
#         R = S - S @ self.B_T @ self.B_T.t()     # [K, D]
#         return U, R

#     def compute_gate(self, U):
#         # [K, 1], initialized to 1 when q = 0
#         gate = 1.0 + torch.tanh(U @ self.q)
#         return gate

#     def forward(self):
#         T = self.base_text_features              # [K, D]
#         U, R = self.compute_residual()           # U: [K, r], R: [K, D]
#         gate = self.compute_gate(U)              # [K, 1]

#         W = T + self.beta * (gate * R)          # broadcast on D
#         return W
    
    
# class NonlinearOrthCompProjLearner(nn.Module):
#     """
#     Nonlinear version of OrthComp-Proj.

#     Provides:
#         base text classifier T
#         residual compensation classifier R
#         old fused classifier T + beta R
#     """

#     def __init__(self, cfg, base_text_features, B_T, B_ref=None, B_con=None, B_R=None, G_init=None):
#         super().__init__()

#         self.beta = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE

#         self.register_buffer("base_text_features", base_text_features.float())
#         self.register_buffer("B_T", B_T.float())

#         if B_ref is not None:
#             self.register_buffer("B_ref", B_ref.float())
#             self.register_buffer("B_con", B_con.float())

#         K, D = base_text_features.shape
#         r = B_T.shape[1]

#         self.linear = nn.Linear(r, D, bias=False)

#         if G_init is not None:
#             with torch.no_grad():
#                 self.linear.weight.copy_(G_init.float().t())
#         else:
#             nn.init.zeros_(self.linear.weight)

#         hidden_dim = min(D, 4 * r)

#         self.delta = nn.Sequential(
#             nn.Linear(r, hidden_dim, bias=True),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, D, bias=False),
#         )

#         nn.init.zeros_(self.delta[-1].weight)

#     def compute_residual(self):
#         T = self.base_text_features
#         U = T @ self.B_T

#         S_linear = self.linear(U)
#         S_delta = self.delta(U)
#         S = S_linear + S_delta

#         # If you want projection w.r.t. reference basis, keep B_ref.
#         # If you want strict current-text orthogonality, replace B_ref by B_T.
#         if hasattr(self, "B_ref"):
#             R = S - (S @ self.B_ref) @ self.B_ref.t()
#         else:
#             R = S - (S @ self.B_T) @ self.B_T.t()

#         return U, R

#     def get_base_text_features(self):
#         return self.base_text_features

#     def get_residual_features(self):
#         _, R = self.compute_residual()
#         return R

#     def get_fused_text_features(self):
#         T = self.base_text_features
#         R = self.get_residual_features()
#         return T + self.beta * R

#     def forward(self):
#         return self.get_fused_text_features()


# # class NonlinearOrthCompProjAdaptiveBetaLearner(nn.Module):
# #     """
# #     Nonlinear OrthComp-Proj with text-conditioned adaptive beta.

# #     Residual generator:
# #         U = T @ B_T
# #         S = Linear(U; G_init) + MLP(U)
# #         R = S - S @ B_T @ B_T^T

# #     Adaptive compensation:
# #         gate = 1 + rho * tanh(gate_net(U))
# #         W = T + beta * gate * R
# #     """

# #     def __init__(self, cfg, base_text_features, B_T, B_R=None, G_init=None):
# #         super().__init__()

# #         self.beta = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE

# #         # gate range: gate in [1-rho, 1+rho]
# #         # For rho=0.5, gate in [0.5, 1.5].
# #         try:
# #             self.gate_rho = float(cfg.TRAINER.OrthComp_adapter.GATE_RHO)
# #         except Exception:
# #             self.gate_rho = 0.5

# #         self.register_buffer("base_text_features", base_text_features.float())  # [K, D]
# #         self.register_buffer("B_T", B_T.float())                                # [D, r]

# #         K, D = base_text_features.shape
# #         r = B_T.shape[1]

# #         # ---------- Linear residual path ----------
# #         self.linear = nn.Linear(r, D, bias=False)

# #         if G_init is not None:
# #             # G_init: [r, D], nn.Linear weight: [D, r]
# #             with torch.no_grad():
# #                 self.linear.weight.copy_(G_init.float().t())
# #         else:
# #             nn.init.zeros_(self.linear.weight)

# #         # ---------- Nonlinear residual path ----------
# #         hidden_dim = min(D, 4 * r)

# #         self.delta = nn.Sequential(
# #             nn.Linear(r, hidden_dim, bias=True),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, D, bias=False),
# #         )

# #         # Make nonlinear branch start from zero.
# #         # Initial model becomes exactly the linear OrthComp-Proj.
# #         nn.init.zeros_(self.delta[-1].weight)

# #         # ---------- Text-conditioned beta gate ----------
# #         try:
# #             gate_hidden_dim = int(cfg.TRAINER.OrthComp_adapter.GATE_HIDDEN_DIM)
# #         except Exception:
# #             gate_hidden_dim = min(128, max(16, 2 * r))

# #         self.gate_net = nn.Sequential(
# #             nn.Linear(r, gate_hidden_dim, bias=True),
# #             nn.ReLU(),
# #             nn.Linear(gate_hidden_dim, 1, bias=True),
# #         )

# #         # Important:
# #         # Make gate_net output zero initially.
# #         # Then tanh(0)=0 and gate=1.
# #         # Therefore the initial model is exactly:
# #         #     W = T + beta * R
# #         nn.init.zeros_(self.gate_net[-1].weight)
# #         nn.init.zeros_(self.gate_net[-1].bias)

# #     def compute_residual(self):
# #         T = self.base_text_features              # [K, D]
# #         U = T @ self.B_T                         # [K, r]

# #         S_linear = self.linear(U)                # [K, D]
# #         S_delta = self.delta(U)                  # [K, D]
# #         S = S_linear + S_delta                   # [K, D]

# #         # Project S onto the orthogonal complement of span(B_T)
# #         R = S - (S @ self.B_T) @ self.B_T.t()    # [K, D]

# #         return U, R

# #     def compute_gate(self, U):
# #         # gate_logits: [K, 1]
# #         gate_logits = self.gate_net(U)

# #         # gate: [K, 1]
# #         # initialized as all ones
# #         gate = 1.0 + self.gate_rho * torch.tanh(gate_logits)

# #         return gate

# #     def forward(self):
# #         T = self.base_text_features              # [K, D]
# #         U, R = self.compute_residual()           # U: [K, r], R: [K, D]

# #         gate = self.compute_gate(U)              # [K, 1]
# #         W = T + self.beta * gate * R             # [K, D]

# #         return W

# #     def gate_regularization(self):
# #         """
# #         Optional regularization.
# #         Keeps adaptive beta close to the original fixed-beta version.
# #         """
# #         T = self.base_text_features
# #         U = T @ self.B_T
# #         gate = self.compute_gate(U)
# #         return (gate - 1.0).pow(2).mean()

# #     def get_gate_values(self):
# #         """
# #         Optional: useful for debugging.
# #         """
# #         with torch.no_grad():
# #             T = self.base_text_features
# #             U = T @ self.B_T
# #             gate = self.compute_gate(U)
# #         return gate.squeeze(-1)



# # # TaskRes-Image
# # class TaskResLearner(nn.Module):
# #     def __init__(self, cfg, classnames, clip_model, base_text_features):
# #         super().__init__()
# #         self.device = clip_model.dtype
# #         # feat_dim = base_text_features.size(-1)
# #         self.alpha = cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE
# #         print(">> DCT scale factor: ", self.alpha)
# #         self.register_buffer("base_text_features", base_text_features)
# #         self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features[0:1]))

# #     def forward(self):
# #         # print(self.base_text_features.dtype, self.text_feature_residuals.dtype)
# #         return self.base_text_features, self.alpha * self.text_feature_residuals

# # def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
# #     device = next(text_encoder.parameters()).device
# #     if clip_model.dtype == torch.float16:
# #         text_encoder = text_encoder.cuda()
    
# #     dataset = cfg.DATASET.NAME

# #     if dataset == "ImageNet":
# #         TEMPLATES = IMAGENET_TEMPLATES_SELECT
# #     else:
# #         TEMPLATES = []
# #     TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

# #     with torch.no_grad():
# #         text_embeddings = []
# #         for text in classnames:
# #             tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
# #             embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
# #             if clip_model.dtype == torch.float16:
# #                 text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
# #             else:
# #                 text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
# #     text_embeddings = torch.stack(text_embeddings).mean(1)
# #     text_encoder = text_encoder.to(device)
# #     return text_embeddings.to(device)

# def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
#     original_device = next(text_encoder.parameters()).device

#     # If CLIP is fp16, text encoder should run on GPU.
#     if clip_model.dtype == torch.float16:
#         text_encoder = text_encoder.cuda()

#     text_device = next(text_encoder.parameters()).device
#     token_device = clip_model.token_embedding.weight.device

#     TEMPLATES = _get_prompt_templates(cfg)

#     print(f">> Use {len(TEMPLATES)} prompt template(s) for {cfg.DATASET.NAME}")
#     for i, temp in enumerate(TEMPLATES):
#         print(f"   [{i}] {temp}")

#     text_encoder.eval()

#     with torch.no_grad():
#         text_features = []

#         for classname in classnames:
#             classname = classname.replace("_", " ")

#             prompts = [template.format(classname) for template in TEMPLATES]

#             # Tokenize on token embedding device
#             tokens = clip.tokenize(prompts).to(token_device)

#             # Token embedding
#             embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)

#             # Move to text encoder device
#             embeddings = embeddings.to(text_device)
#             tokens = tokens.to(text_device)

#             # Encode all prompts of the same class
#             class_features = text_encoder(embeddings, tokens).float()  # [M, D]

#             # Important: normalize each prompt feature first
#             # class_features = F.normalize(class_features, dim=-1)

#             # Then average templates
#             class_feature = class_features.mean(dim=0)

#             # Important: normalize the averaged class feature
#             class_feature = F.normalize(class_feature, dim=0)

#             text_features.append(class_feature)

#         text_features = torch.stack(text_features, dim=0)  # [K, D]

#     text_encoder = text_encoder.to(original_device)

#     return text_features.to(original_device)


# def load_ref_classnames(label_file, exclude_file=None):
#     """
#     Load reference class names from a txt file.

#     Expected formats:
#         1) "goldfish"
#         2) "n01443537 goldfish, Carassius auratus"

#     If a wnid is provided, we keep the first lemma after the wnid.
#     """
#     exclude_wnids = set()
#     exclude_names = set()

#     if exclude_file is not None and osp.exists(exclude_file):
#         with open(exclude_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 item = line.strip()
#                 if not item:
#                     continue
#                 parts = item.split(maxsplit=1)
#                 if parts[0].startswith("n") and parts[0][1:].isdigit():
#                     exclude_wnids.add(parts[0])
#                     if len(parts) > 1:
#                         exclude_names.add(parts[1].split(",")[0].strip().lower())
#                 else:
#                     exclude_names.add(item.lower())

#     classnames = []
#     seen = set()

#     with open(label_file, "r", encoding="utf-8") as f:
#         for line in f:
#             raw = line.strip()
#             if not raw:
#                 continue

#             parts = raw.split(maxsplit=1)

#             wnid = None
#             name_part = raw

#             if parts[0].startswith("n") and parts[0][1:].isdigit() and len(parts) > 1:
#                 wnid = parts[0]
#                 name_part = parts[1]

#             # Keep the first synonym/lemma.
#             name = name_part.split(",")[0].strip()
#             name = name.replace("_", " ")

#             if wnid is not None and wnid in exclude_wnids:
#                 continue
#             if name.lower() in exclude_names:
#                 continue
#             if name.lower() in seen:
#                 continue

#             seen.add(name.lower())
#             classnames.append(name)

#     print(f">> Loaded {len(classnames)} reference class names from {label_file}")
#     return classnames


# def _get_ref_text_features_batched(
#     cfg,
#     ref_classnames,
#     clip_model,
#     text_encoder,
#     batch_size=256,
#     template_text="a photo of a {}."
# ):
#     """
#     Encode reference class names into CLIP text features.

#     Returns:
#         ref_text_features: [N_ref, D]
#     """
#     # text_encoder forward 最终运行在哪个设备
#     text_device = next(text_encoder.parameters()).device

#     # token_embedding 权重所在设备（大概率是 cpu）
#     token_embed_device = clip_model.token_embedding.weight.device

#     # 如果你希望 text encoder 在 GPU 上跑
#     if clip_model.dtype == torch.float16 and text_device.type != "cuda":
#         text_encoder = text_encoder.cuda()
#         text_device = next(text_encoder.parameters()).device

#     all_features = []

#     text_encoder.eval()

#     with torch.no_grad():
#         for start in range(0, len(ref_classnames), batch_size):
#             names = ref_classnames[start:start + batch_size]
#             prompts = [template_text.format(name) for name in names]

#             # 1) tokens 先放到 token_embedding 所在设备
#             tokens = clip.tokenize(prompts).to(token_embed_device)

#             # 2) 在 token_embedding 所在设备上做 lookup
#             embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)

#             # 3) 再把 embeddings 和 tokens 搬到 text_encoder 的设备
#             embeddings = embeddings.to(text_device)
#             tokens = tokens.to(text_device)

#             feats = text_encoder(embeddings, tokens)
#             # feats = feats.float()
#             # feats = feats / feats.norm(dim=-1, keepdim=True)

#             all_features.append(feats.cpu())

#     ref_text_features = torch.cat(all_features, dim=0)

#     print(f">> Reference text features shape: {ref_text_features.shape}")

#     return ref_text_features

# def _get_enhanced_base_text_features(cfg, classnames, clip_model, text_encoder, pretraiend_model):
#     device = next(text_encoder.parameters()).device
#     if clip_model.dtype == torch.float16:
#         text_encoder = text_encoder.cuda()

#         pretrained_text_projection = torch.load(pretraiend_model)

#         state_dict = text_encoder.state_dict()
#         state_dict['text_projection'] = pretrained_text_projection['state_dict']['weight'].t()
#         text_encoder.load_state_dict(state_dict)
#         print(">> Pretrained text encoder loaded!")
#         params = pretrained_text_projection['state_dict']['weight'].size(0) * \
#             pretrained_text_projection['state_dict']['weight'].size(1)
#         print(">> Text projection parameters: ", params)
#         print(pretrained_text_projection['state_dict'].keys())
    
#     dataset = cfg.DATASET.NAME
#     if dataset == "ImageNet":
#         TEMPLATES = IMAGENET_TEMPLATES_SELECT
#     else:
#         TEMPLATES = []
#     TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

#     with torch.no_grad():
#         text_embeddings = []
#         for text in classnames:
#             tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
#             embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
#             if clip_model.dtype == torch.float16:
#                 text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
#             else:
#                 text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
#     text_embeddings = torch.stack(text_embeddings).mean(1)
#     text_encoder = text_encoder.to(device)
#     return text_embeddings.to(device)

# class VisualAdapter(nn.Module):
#     def __init__(self, dim, bottleneck_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, bottleneck_dim, bias=False)
#         self.act = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(bottleneck_dim, dim, bias=False)
#         # self.act2 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # x: [B, D]
#         return self.fc2(self.act(self.fc1(x)))

# class CustomCLIP(nn.Module):
#     def __init__(self, cfg, clip_model, base_text_features, B_T, B_ref, B_con, B_R, G_init=None):
#         super().__init__()
#         self.image_encoder = clip_model.visual
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#         self.prompt_learner = NonlinearOrthCompProjLearner(
#             cfg,
#             base_text_features=base_text_features,
#             B_T=B_T,
#             B_ref=B_ref,
#             B_con=B_con,
#             B_R=B_R,
#             G_init=None,   # important: do not discard G_init
#         )

#         try:
#             self.logit_bias_scale = float(cfg.TRAINER.OrthComp_adapter.LOGIT_BIAS_SCALE)
#         except Exception:
#             self.logit_bias_scale = float(cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE)

#         try:
#             self.normalize_comp_head = bool(cfg.TRAINER.OrthComp_adapter.NORMALIZE_COMP_HEAD)
#         except Exception:
#             self.normalize_comp_head = True
            
#         feat_dim = base_text_features.shape[1]
#         bottleneck_dim = cfg.TRAINER.OrthComp_adapter.VA_BOTTLENECK_DIM
#         self.visual_alpha = cfg.TRAINER.OrthComp_adapter.VA_ALPHA

#         # self.visual_adapter = VisualAdapter(
#         #     dim=feat_dim,
#         #     bottleneck_dim=bottleneck_dim
#         # )

#     def train(self, mode: bool = True):
#         super().train(mode)
#         self.image_encoder.eval()
#         return self

#     def encode_image(self, image):
#         try:
#             image_features = self.image_encoder(image.type(self.dtype))
#         except Exception:
#             image_features = self.image_encoder(image.float())
            
#         # adapted_features = self.visual_adapter(image_features)     # [B, D]

#         # alpha = self.visual_alpha
#         # image_features = (1.0 - alpha) * image_features + alpha * adapted_features

#         image_features = image_features.float()
#         image_features = F.normalize(image_features, dim=-1)
        
#         return image_features

#     def forward(self, image, return_all=False, return_text=False):
#         image_features = self.encode_image(image)

#         # Base text classifier T
#         base_text_features = self.prompt_learner.get_base_text_features().float()
#         base_text_features = F.normalize(base_text_features, dim=-1)

#         # Residual compensation classifier R
#         comp_text_features = self.prompt_learner.get_residual_features().float()

#         if self.normalize_comp_head:
#             comp_text_features = F.normalize(comp_text_features, dim=-1, eps=1e-6)

#         # Logit-level fusion
#         logit_scale = self.logit_scale.exp()

#         logits_base = logit_scale * image_features @ base_text_features.t()
#         logits_comp = logit_scale * image_features @ comp_text_features.t()

#         logits_fused = logits_base + self.logit_bias_scale * logits_comp

#         # Only used for optional regularization / logging
#         fused_text_features = F.normalize(
#             base_text_features + self.logit_bias_scale * comp_text_features,
#             dim=-1,
#             eps=1e-6,
#         )

#         if return_all:
#             return {
#                 "logits_base": logits_base,
#                 "logits_comp": logits_comp,
#                 "logits_fused": logits_fused,
#                 "base_text_features": base_text_features,
#                 "comp_text_features": comp_text_features,
#                 "fused_text_features": fused_text_features,
#             }

#         if return_text:
#             return logits_fused, fused_text_features, base_text_features

#         # Important:
#         # Default output is fused logits, so evaluation uses:
#         # s_final = s_base + beta * s_comp
#         return logits_fused

# @TRAINER_REGISTRY.register()
# class OrthComp_adapter(TrainerX):
#     """Context Optimization (TaskRes).

#     Task Residual for Tuning Vision-Language Models
#     https://arxiv.org/abs/2211.10277
#     """

#     def check_cfg(self, cfg):
#         assert cfg.TRAINER.OrthComp_adapter.PREC in ["fp16", "fp32", "amp"]

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)
        
#         if cfg.TRAINER.OrthComp_adapter.PREC == "fp32" or cfg.TRAINER.OrthComp_adapter.PREC == "amp":
#             # CLIP's default precision is fp16
#             clip_model.float()

#         print("Building custom CLIP")

#         text_encoder = TextEncoder(clip_model)

#         if cfg.TRAINER.OrthComp_adapter.ENHANCED_BASE == "none":
#             print(">> Use regular base!")
#             base_text_features = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
            
#             ref_classnames = load_ref_classnames("ImageNet21K/imagenet21k_classnames_first_lemma.txt")
            
#             ref_cache_path = _get_ref_text_cache_path(cfg)

#             if osp.exists(ref_cache_path) and not cfg.TRAINER.OrthComp_adapter.REF_TEXT_FORCE_REBUILD:
#                 ref_text_features = torch.load(ref_cache_path, map_location="cpu")
#                 print(f">> Loaded reference text feature cache: {ref_cache_path}")
#             else:
#                 ref_text_features = _get_ref_text_features_batched(cfg, ref_classnames, clip_model, text_encoder)
#                 os.makedirs(osp.dirname(ref_cache_path), exist_ok=True)
#                 torch.save(ref_text_features.cpu(), ref_cache_path)
#                 print(f">> Saved reference text feature cache: {ref_cache_path}")
#             # ref_text_features = _get_ref_text_features_batched(cfg, ref_classnames, clip_model, text_encoder)
#         else:
#             print(">> Use enhanced base!")
#             base_text_features = _get_enhanced_base_text_features(
#                 cfg, classnames, clip_model, text_encoder, cfg.TRAINER.OrthComp_adapter.ENHANCED_BASE
#             )


#         print(f">> Base text features shape: {base_text_features.shape}")
#         # self.model = CustomCLIP(cfg, classnames, clip_model)

#         # Step 1: build frozen image feature cache for later probe fitting
#         if cfg.TRAINER.OrthComp_adapter.BUILD_FEAT_CACHE:
#             cache_path = _get_feature_cache_path(cfg)

#             if osp.exists(cache_path) and not cfg.TRAINER.OrthComp_adapter.FEAT_CACHE_FORCE_REBUILD:
#                 print(f">> Found existing feature cache: {cache_path}")
#                 feats, labels = load_feature_cache(cache_path)
#             else:
#                 print(">> Building training feature cache from frozen CLIP image encoder...")
#                 image_encoder = clip_model.visual.to(self.device)
#                 image_encoder.eval()

#                 feats, labels = extract_image_features(
#                     data_loader=self.train_loader_x,
#                     image_encoder=image_encoder,
#                     device=self.device,
#                     dtype=clip_model.dtype,
#                     normalize=True
#                 )
#                 save_feature_cache(cache_path, feats, labels)
#                 print(f">> Feature cache saved to: {cache_path}")

#                 print("len(train_loader_x):", len(self.train_loader_x))              # batch 数
#                 print("len(train_loader_x.dataset):", len(self.train_loader_x.dataset))  # 数据集样本数
#                 print("extracted feature num:", feats.shape[0])                      # 实际提取样本数

#             print(f">> Cached train features shape: {feats.shape}")
#             print(f">> Cached train labels shape: {labels.shape}")
            
#             print(">> Building class prototypes from cached frozen image features...")
#             train_class_prototypes, proto_counts = build_class_prototypes_from_features(
#                 feats=feats,
#                 labels=labels,
#                 num_classes=self.num_classes,
#                 normalize=True,
#             )

#             print(f">> Train class prototypes shape: {train_class_prototypes.shape}")
#             print(f">> Prototype counts: min={proto_counts.min().item():.0f}, max={proto_counts.max().item():.0f}")

#             self.train_class_prototypes = train_class_prototypes

#         # Step 2: fit/load surrogate probe on frozen image features
#         if cfg.TRAINER.OrthComp_adapter.BUILD_PROBE:
#             probe_path = _get_probe_cache_path(cfg)

#             if osp.exists(probe_path) and not cfg.TRAINER.OrthComp_adapter.PROBE_FORCE_REBUILD:
#                 print(f">> Found existing probe cache: {probe_path}")
#                 probe_weight, probe_acc, probe_meta = load_probe_cache(probe_path)
#             else:
#                 print(">> Fitting surrogate probe on frozen CLIP image features...")
#                 probe_weight, probe_acc = fit_surrogate_probe(
#                     train_feats=feats,
#                     train_labels=labels,
#                     num_classes=self.num_classes,
#                     lr=cfg.TRAINER.OrthComp_adapter.PROBE_LR,
#                     weight_decay=cfg.TRAINER.OrthComp_adapter.PROBE_WD,
#                     epochs=cfg.TRAINER.OrthComp_adapter.PROBE_EPOCHS,
#                     batch_size=cfg.TRAINER.OrthComp_adapter.PROBE_BATCH_SIZE,
#                     device=self.device,
#                     bias=False,
#                     init_mode=cfg.TRAINER.OrthComp_adapter.PROBE_INIT,
#                 )

#                 probe_meta = {
#                     "dataset": cfg.DATASET.NAME,
#                     "backbone": cfg.MODEL.BACKBONE.NAME,
#                     "num_shots": cfg.DATASET.NUM_SHOTS,
#                     "seed": cfg.SEED,
#                     "num_classes": self.num_classes,
#                     "feat_dim": feats.size(1),
#                 }
#                 save_probe_cache(probe_path, probe_weight, probe_acc, probe_meta)
#                 print(f">> Probe cache saved to: {probe_path}")

#             print(f">> Probe weight shape: {probe_weight.shape}")
#             if probe_acc is not None:
#                 print(f">> Probe cached train acc: {probe_acc:.2f}%")

#             # keep for later subspace construction
#             self.probe_weight = probe_weight

#         # Step 3: build/load text basis and residual basis
#         if cfg.TRAINER.OrthComp_adapter.BUILD_SUBSPACE:
#             subspace_path = _get_subspace_cache_path(cfg)

#             if osp.exists(subspace_path) and not cfg.TRAINER.OrthComp_adapter.SUBSPACE_FORCE_REBUILD:
#                 print(f">> Found existing subspace cache: {subspace_path}")
#                 subspace_data = load_subspace_cache(subspace_path)
#                 B_T = subspace_data["B_T"]
#                 B_R = subspace_data["B_R"]
#                 text_rank = subspace_data["text_rank"]
#                 residual_singular_values = subspace_data["residual_singular_values"]
#             else:
#                 print(">> Building text subspace basis B_T ...")
                
#                 B_ref, text_singular_values, text_rank = build_reference_text_basis(
#                     ref_text_features=ref_text_features.to(self.device),
#                     rank=cfg.TRAINER.OrthComp_adapter.REF_BASIS_RANK,
#                     center=cfg.TRAINER.OrthComp_adapter.REF_BASIS_CENTER
#                 )

#                 B_T, text_singular_values, text_rank = build_text_basis(base_text_features, rank=cfg.TRAINER.OrthComp_adapter.TEXT_BASIS_RANK)
                
#                 B_con = torch.cat((B_T, B_ref), dim=1)

#                 print(">> Computing probe residual outside text subspace ...")
#                 W_res = compute_probe_residual(self.probe_weight, B_T)

#                 print(">> Building low-rank residual basis B_R ...")
#                 B_R, residual_singular_values, selected_rank, cumulative_energy = build_residual_basis(
#                     W_res,
#                     k=cfg.TRAINER.OrthComp_adapter.RESIDUAL_RANK,
#                     energy_thresh=cfg.TRAINER.OrthComp_adapter.ENERGY_THRESH,
#                     auto_rank=cfg.TRAINER.OrthComp_adapter.AUTO_RANK,
#                     min_rank=cfg.TRAINER.OrthComp_adapter.MIN_RANK,
#                     max_rank=cfg.TRAINER.OrthComp_adapter.MAX_RANK
#                 )
#                 print(f">> Selected residual rank k = {selected_rank}")
#                 print(f">> Top-10 cumulative energy = {cumulative_energy[:10].cpu().numpy()}")
#                 self.selected_rank = selected_rank
#                 self.B_R = B_R

#                 save_subspace_cache(
#                     subspace_path,
#                     B_T=B_T,
#                     B_R=B_R,
#                     text_rank=text_rank,
#                     residual_singular_values=residual_singular_values,
#                     selected_rank=selected_rank,
#                     cumulative_energy=cumulative_energy,
#                     meta={
#                         "dataset": cfg.DATASET.NAME,
#                         "backbone": cfg.MODEL.BACKBONE.NAME,
#                         "num_shots": cfg.DATASET.NUM_SHOTS,
#                         "seed": cfg.SEED,
#                         "text_feat_shape": list(base_text_features.shape),
#                         "probe_weight_shape": list(self.probe_weight.shape),
#                     }
#                 )
#                 print(f">> Subspace cache saved to: {subspace_path}")

#             print(f">> B_T shape: {B_T.shape}, text rank: {text_rank}")
#             print(f">> B_R shape: {B_R.shape}")
#             self.B_T = B_T
#             self.B_R = B_R
#             self.B_ref = B_ref
#             self.B_con = B_con
#             # orth_check = torch.norm(B_T.t() @ B_R).item()
#             # print(f">> ||B_T^T B_R||_F = {orth_check:.6e}")
#             # print(f">> Residual singular values (top 10): {residual_singular_values[:10]}")

#         G_init = solve_ridge_init_G(
#             base_text_features=base_text_features,
#             B_T=self.B_T,
#             ref_weight=self.probe_weight,
#             ridge_lambda=cfg.TRAINER.OrthComp_adapter.RIDGE_INIT_LAMBDA
#         )

#         self.model = CustomCLIP(
#                 cfg,
#                 clip_model,
#                 base_text_features=base_text_features,
#                 B_T=self.B_T,
#                 B_ref = self.B_ref,
#                 B_con = self.B_con,
#                 B_R=self.B_R,
#                 G_init=G_init
#             )

#         print("Turning off gradients in CLIP encoders, keeping prompt_learner and visual_adapter trainable")
#         for name, param in self.model.named_parameters():
#             if ("prompt_learner" not in name) and ("visual_adapter" not in name):
#                 param.requires_grad_(False)
#             else:
#                 print(name)

#         if cfg.MODEL.INIT_WEIGHTS:
#             load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

#         self.model.to(self.device)
#         self.model = self.model.float()
        
#         if hasattr(self, "train_class_prototypes"):
#             self.train_class_prototypes = self.train_class_prototypes.to(self.device)
            
            
#         # NOTE: only give prompt_learner to the optimizer
#         trainable_params = []
#         trainable_params += list(self.model.prompt_learner.parameters())
#         # trainable_params += list(self.model.visual_adapter.parameters())
#         self.optim = build_optimizer(self.model, cfg.OPTIM, trainable_params)
#         self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
#         self.register_model("model", self.model, self.optim, self.sched)

#         self.scaler = GradScaler() if cfg.TRAINER.OrthComp_adapter.PREC == "amp" else None
        
        
#         try:
#             self.proto_loss_weight = float(cfg.TRAINER.OrthComp_adapter.PROTO_LOSS_WEIGHT)
#         except Exception:
#             self.proto_loss_weight = 0.0

#         try:
#             self.proto_logit_scale = float(cfg.TRAINER.OrthComp_adapter.PROTO_LOGIT_SCALE)
#         except Exception:
#             self.proto_logit_scale = 20.0

#         print(f">> Prototype CE weight: {self.proto_loss_weight}")
#         print(f">> Prototype CE logit scale: {self.proto_logit_scale}")
        
#         try:
#             self.geo_loss_weight = float(cfg.TRAINER.OrthComp_adapter.GEO_LOSS_WEIGHT)
#         except Exception:
#             self.geo_loss_weight = 0.0
        
#         try:
#             self.geo_topk = int(cfg.TRAINER.OrthComp_adapter.GEO_TOPK)
#         except Exception:
#             self.geo_topk = 5

#         try:
#             self.geo_margin = float(cfg.TRAINER.OrthComp_adapter.GEO_MARGIN)
#         except Exception:
#             self.geo_margin = 0.0

#         print(f">> Classifier geometry loss weight: {self.geo_loss_weight}")
#         print(f">> Classifier geometry top-k: {self.geo_topk}")
#         print(f">> Classifier geometry margin: {self.geo_margin}")
        
#         try:
#             self.geo_tau = float(cfg.TRAINER.OrthComp_adapter.GEO_TAU)
#         except Exception:
#             self.geo_tau = 0.05

#         try:
#             self.geo_use_w_topk = bool(cfg.TRAINER.OrthComp_adapter.GEO_USE_W_TOPK)
#         except Exception:
#             self.geo_use_w_topk = False

#         print(f">> Classifier geometry tau: {self.geo_tau}")
#         print(f">> Use W-topk geometry neighbors: {self.geo_use_w_topk}")
        
#         try:
#             self.train_loss_mode = str(cfg.TRAINER.OrthComp_adapter.TRAIN_LOSS_MODE)
#         except Exception:
#             self.train_loss_mode = "comp_only"

#         try:
#             self.multibranch_lambda = float(cfg.TRAINER.OrthComp_adapter.MULTIBRANCH_LAMBDA)
#         except Exception:
#             self.multibranch_lambda = 0.5

#         print(f">> Train loss mode: {self.train_loss_mode}")
#         print(f">> Multi-branch lambda: {self.multibranch_lambda}")

#         # Note that multi-gpu training could be slow because CLIP's size is
#         # big, which slows down the copy operation in DataParallel
#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)

#     def forward_backward(self, batch):
#         image, label = self.parse_batch_train(batch)
#         prec = self.cfg.TRAINER.OrthComp_adapter.PREC

#         out = self.model(image, return_all=True)

#         logits_base = out["logits_base"]
#         logits_comp = out["logits_comp"]
#         logits_fused = out["logits_fused"]

#         loss_base = F.cross_entropy(logits_base, label)
#         loss_comp = F.cross_entropy(logits_comp, label)
#         loss_fusion = F.cross_entropy(logits_fused, label)

#         if self.train_loss_mode == "comp_only":
#             loss_ce = loss_comp
#         elif self.train_loss_mode == "fusion_only":
#             loss_ce = loss_fusion
#         elif self.train_loss_mode == "multi_branch":
#             lam = self.multibranch_lambda
#             loss_ce = (1 - self.multibranch_lambda) * loss_fusion + self.multibranch_lambda * loss_comp
#         else:
#             raise ValueError(f"Unknown TRAIN_LOSS_MODE: {self.train_loss_mode}")

#         text_for_reg = out["fused_text_features"]

#         loss_proto = self.compute_proto_ce_loss(text_for_reg)
#         loss_geo = self.compute_classifier_absolute_separation_loss(
#             text_features=text_for_reg
#         )

#         loss = (
#             loss_ce
#             + self.proto_loss_weight * loss_proto
#             + self.geo_loss_weight * loss_geo
#         )

#         self.model_backward_and_update(loss)

#         loss_summary = {
#             "loss": loss.item(),
#             "loss_ce": loss_ce.item(),
#             # "loss_comp": loss_comp.item(),
#             # "loss_fusion": loss_fusion.item(),
#             # "loss_proto": loss_proto.item(),
#             # "loss_geo": loss_geo.item(),
#             "acc": compute_accuracy(logits_fused, label)[0].item(),
#             # "acc_base": compute_accuracy(logits_base, label)[0].item(),
#             "acc_comp": compute_accuracy(logits_comp, label)[0].item(),
#             # "acc_fused": compute_accuracy(logits_fused, label)[0].item(),
#         }

#         if (self.batch_idx + 1) == self.num_batches:
#             self.update_lr()

#         return loss_summary
    
#     def compute_proto_ce_loss(self, text_features):
#         """
#         Prototype CE:
#             prototypes are used as queries,
#             generated text classifiers are used as class weights.

#         Args:
#             text_features: [C, D], normalized generated classifiers

#         Returns:
#             loss_proto: scalar
#         """
#         if (not hasattr(self, "train_class_prototypes")) or self.proto_loss_weight <= 0:
#             return text_features.new_tensor(0.0)

#         prototypes = self.train_class_prototypes.to(text_features.device)
#         prototypes = F.normalize(prototypes.float(), dim=-1)
#         text_features = F.normalize(text_features.float(), dim=-1)

#         logits_proto = self.proto_logit_scale * prototypes @ text_features.t()  # [C, C]
#         labels_proto = torch.arange(text_features.size(0), device=text_features.device)

#         loss_proto = F.cross_entropy(logits_proto, labels_proto)
#         return loss_proto
    
#     def compute_classifier_absolute_separation_loss(self, text_features):
#         """
#         Absolute separation loss:
#         directly penalize large pairwise similarity among W.
#         """
#         if self.geo_loss_weight <= 0:
#             return text_features.new_tensor(0.0)

#         W = F.normalize(text_features.float(), dim=-1)

#         C = W.size(0)
#         if C <= 1:
#             return text_features.new_tensor(0.0)

#         sim_w = W @ W.t()

#         eye = torch.eye(C, device=W.device, dtype=torch.bool)

#         k = min(self.geo_topk, C - 1)

#         sim_w_for_topk = sim_w.detach().masked_fill(eye, -1e4)
#         _, idx_w = torch.topk(sim_w_for_topk, k=k, dim=1)

#         row_idx = torch.arange(C, device=W.device).unsqueeze(1).expand(C, k)

#         hard_w = sim_w[row_idx, idx_w]

#         # target maximum similarity
#         target_sim = getattr(self, "geo_target_sim", 0.5)

#         violation = hard_w - target_sim

#         loss_sep = self.geo_tau * F.softplus(violation / self.geo_tau).mean()

#         return loss_sep

#     def parse_batch_train(self, batch):
#         input = batch["img"]
#         label = batch["label"]
#         input = input.to(self.device)
#         label = label.to(self.device)
#         return input, label

#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print("Note that load_model() is skipped as no pretrained model is given")
#             return

#         names = self.get_model_names()

#         # By default, the best model is loaded
#         model_file = "model-best.pth.tar"

#         if epoch is not None:
#             model_file = "model.pth.tar-" + str(epoch)

#         for name in names:
#             model_path = osp.join(directory, name, model_file)

#             if not osp.exists(model_path):
#                 raise FileNotFoundError('Model not found at "{}"'.format(model_path))

#             checkpoint = load_checkpoint(model_path)
#             state_dict = checkpoint["state_dict"]

#             if self.cfg.DATASET.NAME == 'ImageNetA' or self.cfg.DATASET.NAME == 'ImageNetR':
#                 if self.cfg.DATASET.NAME == 'ImageNetA':
#                     from .imagenet_a_r_indexes_v2 import find_imagenet_a_indexes as find_indexes
#                 else:
#                     from .imagenet_a_r_indexes_v2 import find_imagenet_r_indexes as find_indexes
#                 imageneta_indexes = find_indexes()
#                 state_dict['base_text_features'] = state_dict['base_text_features'][imageneta_indexes]
#                 state_dict['text_feature_residuals'] = state_dict['text_feature_residuals'][imageneta_indexes]

#             epoch = checkpoint["epoch"]

#             # Ignore fixed token vectors
#             if "token_prefix" in state_dict:
#                 del state_dict["token_prefix"]

#             if "token_suffix" in state_dict:
#                 del state_dict["token_suffix"]

#             print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
#             # set strict=False
#             self._models[name].load_state_dict(state_dict, strict=False)
