import argparse
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTHONNOUSERSITE"] = "1"

sys.path = [
    p for p in sys.path
    if "/home/jincai_guo/.local/lib/python3.9/site-packages" not in p
]
print(sys.executable)
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.imagenet_a_filter
import datasets.imagenet_r_filter

# import trainers.taskres
# import trainers.orthcomp
import trainers.orthcomp_adapter
import trainers.zsclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # cfg.TRAINER.TaskRes = CN()
    # cfg.TRAINER.TaskRes.N_CTX = 16  # number of context vectors
    # cfg.TRAINER.TaskRes.CSC = False  # class-specific context
    # cfg.TRAINER.TaskRes.CTX_INIT = ""  # initialization words
    # cfg.TRAINER.TaskRes.PREC = "fp16"  # fp16, fp32, amp
    # cfg.TRAINER.TaskRes.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    # cfg.TRAINER.TaskRes.RESIDUAL_SCALE = 1.0
    # cfg.TRAINER.TaskRes.ENHANCED_BASE = args.enhanced_base
    # cfg.TRAINER.TaskRes.BUILD_FEAT_CACHE = True
    # cfg.TRAINER.TaskRes.FEAT_CACHE_DIR = "feature_cache"
    # cfg.TRAINER.TaskRes.FEAT_CACHE_FORCE_REBUILD = True
    # cfg.TRAINER.TaskRes.BUILD_PROBE = True
    # cfg.TRAINER.TaskRes.PROBE_DIR = "probe_cache"
    # cfg.TRAINER.TaskRes.PROBE_FORCE_REBUILD = True
    # cfg.TRAINER.TaskRes.PROBE_EPOCHS = 100
    # cfg.TRAINER.TaskRes.PROBE_LR = 1e-2
    # cfg.TRAINER.TaskRes.PROBE_WD = 1e-4
    # cfg.TRAINER.TaskRes.PROBE_BATCH_SIZE = 256
    # cfg.TRAINER.TaskRes.BUILD_SUBSPACE = True
    # cfg.TRAINER.TaskRes.SUBSPACE_DIR = "subspace_cache"
    # cfg.TRAINER.TaskRes.SUBSPACE_FORCE_REBUILD = True
    # cfg.TRAINER.TaskRes.RESIDUAL_RANK = 16


    # cfg.TRAINER.OrthComp = CN()
    # cfg.TRAINER.OrthComp.N_CTX = 16  # number of context vectors
    # cfg.TRAINER.OrthComp.CSC = False  # class-specific context
    # cfg.TRAINER.OrthComp.PREC = "fp16"
    # cfg.TRAINER.OrthComp.RESIDUAL_SCALE = 1.0
    # cfg.TRAINER.OrthComp.ENHANCED_BASE = args.enhanced_base
    # cfg.TRAINER.OrthComp.BUILD_FEAT_CACHE = True
    # cfg.TRAINER.OrthComp.FEAT_CACHE_DIR = "feature_cache"
    # cfg.TRAINER.OrthComp.FEAT_CACHE_FORCE_REBUILD = True
    # cfg.TRAINER.OrthComp.BUILD_PROBE = True
    # cfg.TRAINER.OrthComp.PROBE_DIR = "probe_cache"
    # cfg.TRAINER.OrthComp.PROBE_FORCE_REBUILD = True
    # cfg.TRAINER.OrthComp.PROBE_EPOCHS = 100
    # cfg.TRAINER.OrthComp.PROBE_LR = 1e-2
    # cfg.TRAINER.OrthComp.PROBE_WD = 1e-4
    # cfg.TRAINER.OrthComp.PROBE_BATCH_SIZE = 256
    # cfg.TRAINER.OrthComp.BUILD_SUBSPACE = True
    # cfg.TRAINER.OrthComp.SUBSPACE_DIR = "subspace_cache"
    # cfg.TRAINER.OrthComp.SUBSPACE_FORCE_REBUILD = True
    # cfg.TRAINER.OrthComp.RESIDUAL_RANK = 32
    # cfg.TRAINER.OrthComp.AUTO_RANK = True
    # cfg.TRAINER.OrthComp.ENERGY_THRESH = 0.95
    # cfg.TRAINER.OrthComp.MIN_RANK = 4
    # cfg.TRAINER.OrthComp.MAX_RANK = 128
    # cfg.TRAINER.OrthComp.PROBE_INIT = "random"
    # cfg.TRAINER.OrthComp.LAMBDA_C = 1e-2
    # cfg.TRAINER.OrthComp.USE_RIDGE_INIT = True
    # cfg.TRAINER.OrthComp.RIDGE_INIT_LAMBDA = 1e-3

    # cfg.TRAINER.OrthComp_adapter = CN()
    # cfg.TRAINER.OrthComp_adapter.N_CTX = 16  # number of context vectors
    # cfg.TRAINER.OrthComp_adapter.CSC = False  # class-specific context
    # cfg.TRAINER.OrthComp_adapter.PREC = "fp16"
    # cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE = 0.05
    # cfg.TRAINER.OrthComp_adapter.ENHANCED_BASE = args.enhanced_base
    # cfg.TRAINER.OrthComp_adapter.BUILD_FEAT_CACHE = True
    # cfg.TRAINER.OrthComp_adapter.FEAT_CACHE_DIR = "feature_cache"
    # cfg.TRAINER.OrthComp_adapter.FEAT_CACHE_FORCE_REBUILD = True
    # cfg.TRAINER.OrthComp_adapter.BUILD_PROBE = True
    # cfg.TRAINER.OrthComp_adapter.PROBE_DIR = "probe_cache"
    # cfg.TRAINER.OrthComp_adapter.PROBE_FORCE_REBUILD = True
    # cfg.TRAINER.OrthComp_adapter.PROBE_EPOCHS = 100
    # cfg.TRAINER.OrthComp_adapter.PROBE_LR = 1e-2
    # cfg.TRAINER.OrthComp_adapter.PROBE_WD = 1e-4
    # cfg.TRAINER.OrthComp_adapter.PROBE_BATCH_SIZE = 256
    # cfg.TRAINER.OrthComp_adapter.BUILD_SUBSPACE = True
    # cfg.TRAINER.OrthComp_adapter.SUBSPACE_DIR = "subspace_cache"
    # cfg.TRAINER.OrthComp_adapter.SUBSPACE_FORCE_REBUILD = True
    # cfg.TRAINER.OrthComp_adapter.RESIDUAL_RANK = 32
    # cfg.TRAINER.OrthComp_adapter.AUTO_RANK = True
    # cfg.TRAINER.OrthComp_adapter.ENERGY_THRESH = 0.95
    # cfg.TRAINER.OrthComp_adapter.MIN_RANK = 4
    # cfg.TRAINER.OrthComp_adapter.MAX_RANK = 128
    # cfg.TRAINER.OrthComp_adapter.PROBE_INIT = "random"
    # cfg.TRAINER.OrthComp_adapter.LAMBDA_C = 1e-2
    # cfg.TRAINER.OrthComp_adapter.USE_RIDGE_INIT = True
    # cfg.TRAINER.OrthComp_adapter.RIDGE_INIT_LAMBDA = 1e-3
    # cfg.TRAINER.OrthComp_adapter.VA_BOTTLENECK_DIM = 512
    # cfg.TRAINER.OrthComp_adapter.VA_ALPHA = 0.
    # cfg.TRAINER.OrthComp_adapter.TEXT_BASIS_RANK = 128
    # cfg.TRAINER.OrthComp_adapter.REF_BASIS_RANK = 128
    # cfg.TRAINER.OrthComp_adapter.REF_BASIS_CENTER=False
    # cfg.TRAINER.OrthComp_adapter.REF_TEXT_CACHE_DIR="ref_text_cache"
    # cfg.TRAINER.OrthComp_adapter.REF_TEXT_FORCE_REBUILD = False
    # cfg.TRAINER.OrthComp_adapter.PROMPT_ENSEMBLE = True
    # cfg.TRAINER.OrthComp_adapter.PROTO_LOSS_WEIGHT = 0.
    # cfg.TRAINER.OrthComp_adapter.PROTO_LOGIT_SCALE = 20
    # cfg.TRAINER.OrthComp_adapter.GEO_TOPK = 5
    # cfg.TRAINER.OrthComp_adapter.GEO_MARGIN = 0.05
    # cfg.TRAINER.OrthComp_adapter.GEO_LOSS_WEIGHT = 0.
    # cfg.TRAINER.OrthComp_adapter.GEO_TAU = 0.05
    # cfg.TRAINER.OrthComp_adapter.GEO_USE_W_TOPK = True
    # cfg.TRAINER.OrthComp_adapter.TRAIN_LOSS_MODE = "multi_branch"
    # cfg.TRAINER.OrthComp_adapter.LOGIT_BIAS_SCALE = 0.2
    # cfg.TRAINER.OrthComp_adapter.NORMALIZE_COMP_HEAD = False
    # cfg.TRAINER.OrthComp_adapter.MULTIBRANCH_LAMBDA = 0.2
    # cfg.TRAINER.OrthComp_adapter.PROTO_LOSS_WEIGHT = 0.0
    # cfg.TRAINER.OrthComp_adapter.GEO_LOSS_WEIGHT = 0.0
    
    # cfg.TRAINER.OrthComp_adapter.PB_COND_BASIS = "bt"
    # # cfg.TRAINER.OrthComp_adapter.LR_RESIDUAL_RANK = 128
    # cfg.TRAINER.OrthComp_adapter.PB_HIDDEN_DIM = 128
    # cfg.TRAINER.OrthComp_adapter.PB_USE_GATE = False
    # cfg.TRAINER.OrthComp_adapter.PB_GATE_RHO = 0.5
    # cfg.TRAINER.OrthComp_adapter.PB_USE_TANH_COEFF = False
    # # cfg.TRAINER.OrthComp_adapter.LR_NORMALIZE_A = True
    # cfg.TRAINER.OrthComp_adapter.PB_BASIS_TYPE = "residual_proto"
    # cfg.TRAINER.OrthComp_adapter.PB_NORMALIZE_BASIS = True
    
    cfg.TRAINER.OrthComp_adapter = CN()

    # precision
    cfg.TRAINER.OrthComp_adapter.PREC = "fp16"   # "fp16", "fp32", or "amp"

    # main compensation strength
    cfg.TRAINER.OrthComp_adapter.RESIDUAL_SCALE = 1.0

    # prompt setting
    cfg.TRAINER.OrthComp_adapter.PROMPT_ENSEMBLE = True

    # text feature construction
    # False = keep the previous behavior; True = normalize each prompt feature before averaging
    cfg.TRAINER.OrthComp_adapter.NORMALIZE_TEXT_FEATURES = True

    # text-coordinate basis B_T
    cfg.TRAINER.OrthComp_adapter.TEXT_BASIS_RANK = 1024
    cfg.TRAINER.OrthComp_adapter.TEXT_PROJ_RANK = 4
    cfg.TRAINER.OrthComp_adapter.TEXT_BASIS_CENTER = False

    # nonlinear text-conditioned compensator
    cfg.TRAINER.OrthComp_adapter.HIDDEN_DIM = 128
    # 256 for ResNet50

    # lightweight optional switches for future ablation
    cfg.TRAINER.OrthComp_adapter.USE_LAYER_NORM = False
    cfg.TRAINER.OrthComp_adapter.PROJECT_RESIDUAL = True
    
    cfg.TRAINER.OrthComp_adapter.CROSS_DATASET_EVAL = False



    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "base"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # # 4. From optional input arguments
    # cfg.merge_from_list(args.opts)

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.DATASET.NUM_SHOTS = args.num_shots
    
    # 新增：把 eval checkpoint 信息写入 cfg，供 build_model 阶段读取 source B_T
    if args.eval_only and args.model_dir:
        cfg.TRAINER.OrthComp_adapter.SOURCE_MODEL_DIR = args.model_dir

    if args.eval_only and args.load_epoch is not None:
        cfg.TRAINER.OrthComp_adapter.SOURCE_LOAD_EPOCH = args.load_epoch

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, default="/nvme/data-pool1/jincai_guo/tianqi", help="path to dataset")
    parser.add_argument("--root", type=str, default="DATA", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/FINAL/debug/food101/VIT_adapter_base_1.0_100_128_4_with/seed3", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=3, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/TaskRes/adapter.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/food101.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="OrthComp_adapter", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", default=False, help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, default=50, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=16,
        help="number of shots for each class"
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="residual scaling factor for TaskRes"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--enhanced-base", type=str, default="none", help="path to enhanced base classifier weight"
    )

    args = parser.parse_args()
    main(args)
