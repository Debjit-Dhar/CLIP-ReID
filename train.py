from utils.logger import setup_logger
from datasets.make_dataloader import make_dataloader
from model.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg_base as cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_base.yml", help="path to config file", type=str
    )

    parser.add_argument("--use_dicma", action="store_true", help="Enable DiCMA losses")
    parser.add_argument("--dicma_alpha", type=float, default=None, help="W2 loss weight")
    parser.add_argument("--dicma_beta", type=float, default=None, help="Covariance loss weight")
    parser.add_argument("--dicma_gamma", type=float, default=None, help="Relational GW loss weight")
    parser.add_argument("--dicma_rank", type=int, default=None, help="Projected dimension for DiCMA covariance")
    parser.add_argument("--dicma_ema", type=float, default=None, help="EMA momentum for DiCMA running stats")
    parser.add_argument("--dicma_use_gw", action="store_true", help="Enable relational-GW loss term")
    parser.add_argument("--dicma_use_overlapping_patches", action="store_true", help="Use overlapping patches for DiCMA")
    parser.add_argument("--dicma_num_patches", type=int, default=None, help="Number of patches to sample")
    parser.add_argument("--dicma_patch_size", type=int, default=None, help="Patch size for overlapping patches")
    parser.add_argument("--dicma_patch_stride", type=int, default=None, help="Stride for overlapping patches")
    parser.add_argument("--dicma_use_side_embedding", action="store_true", help="Use side embeddings")
    parser.add_argument("--dicma_use_rerank", action="store_true", help="Enable reranking during evaluation")
    parser.add_argument("--dicma_rerank_k1", type=int, default=None, help="Reranking k1 parameter")
    parser.add_argument("--dicma_rerank_k2", type=int, default=None, help="Reranking k2 parameter")
    parser.add_argument("--dicma_rerank_lambda", type=float, default=None, help="Reranking lambda parameter")

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # CLI overrides for DiCMA
    if args.use_dicma:
        cfg.DICMA.ENABLED = True
    if args.dicma_alpha is not None:
        cfg.DICMA.ALPHA = args.dicma_alpha
    if args.dicma_beta is not None:
        cfg.DICMA.BETA = args.dicma_beta
    if args.dicma_gamma is not None:
        cfg.DICMA.GAMMA = args.dicma_gamma
    if args.dicma_rank is not None:
        cfg.DICMA.RANK = args.dicma_rank
    if args.dicma_ema is not None:
        cfg.DICMA.EMA_MOMENTUM = args.dicma_ema
    if args.dicma_use_gw:
        cfg.DICMA.USE_GW = True
    if args.dicma_use_overlapping_patches:
        cfg.DICMA.USE_OVERLAPPING_PATCHES = True
    if args.dicma_num_patches is not None:
        cfg.DICMA.NUM_PATCHES = args.dicma_num_patches
    if args.dicma_patch_size is not None:
        cfg.DICMA.PATCH_SIZE = args.dicma_patch_size
    if args.dicma_patch_stride is not None:
        cfg.DICMA.PATCH_STRIDE = args.dicma_patch_stride
    if args.dicma_use_side_embedding:
        cfg.DICMA.USE_SIDE_EMBEDDING = True
    if args.dicma_use_rerank:
        cfg.DICMA.USE_RERANK = True
    if args.dicma_rerank_k1 is not None:
        cfg.DICMA.RERANK_K1 = args.dicma_rerank_k1
    if args.dicma_rerank_k2 is not None:
        cfg.DICMA.RERANK_K2 = args.dicma_rerank_k2
    if args.dicma_rerank_lambda is not None:
        cfg.DICMA.RERANK_LAMBDA = args.dicma_rerank_lambda

    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    # DiCMA Gaussian prototypes (optional)
    dicma_module = None
    if cfg.DICMA.ENABLED:
        from dicma import GaussianPrototypes

        # Choose feature dimension for DiCMA (before BN if available)
        feat_dim = getattr(model, 'in_planes', None) or getattr(model, 'in_planes_proj', None)
        if feat_dim is None:
            feat_dim = 2048

        # Calculate side embedding dimension
        side_embed_dim = 0
        if cfg.MODEL.SIE_CAMERA:
            side_embed_dim += 1
        if cfg.MODEL.SIE_VIEW:
            side_embed_dim += 1
        if side_embed_dim == 0:
            side_embed_dim = 2  # default fallback

        dicma_module = GaussianPrototypes(
            num_ids=num_classes,
            feat_dim=feat_dim,
            rank=cfg.DICMA.RANK,
            eps=cfg.DICMA.EPS,
            ema_momentum=cfg.DICMA.EMA_MOMENTUM,
            use_relational_gw=cfg.DICMA.USE_GW,
            use_overlapping_patches=cfg.DICMA.USE_OVERLAPPING_PATCHES,
            num_patches=cfg.DICMA.NUM_PATCHES,
            patch_size=cfg.DICMA.PATCH_SIZE,
            patch_stride=cfg.DICMA.PATCH_STRIDE,
            use_side_embedding=cfg.DICMA.USE_SIDE_EMBEDDING,
            side_embed_dim=side_embed_dim,
        )
        dicma_module.to('cuda')

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    if dicma_module is not None:
        optimizer.add_param_group({
            "params": dicma_module.parameters(),
            "lr": cfg.DICMA.LR,
            "weight_decay": cfg.DICMA.WEIGHT_DECAY,
        })

    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
        dicma_module=dicma_module,
    )
