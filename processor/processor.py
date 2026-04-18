import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query,
             local_rank,
             dicma_module=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    dicma_w2_meter = AverageMeter() if cfg.DICMA.ENABLED else None
    dicma_cov_meter = AverageMeter() if cfg.DICMA.ENABLED else None
    dicma_gw_meter = AverageMeter() if cfg.DICMA.ENABLED else None
    dicma_rep_meter = AverageMeter() if cfg.DICMA.ENABLED else None

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                           reranking=cfg.DICMA.USE_RERANK,
                           rerank_k1=cfg.DICMA.RERANK_K1,
                           rerank_k2=cfg.DICMA.RERANK_K2,
                           rerank_lambda=cfg.DICMA.RERANK_LAMBDA)
    scaler = amp.GradScaler()
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        if dicma_module is not None and cfg.DICMA.ENABLED:
            dicma_w2_meter.reset()
            dicma_cov_meter.reset()
            dicma_gw_meter.reset()
            dicma_rep_meter.reset()
        evaluator.reset()

        # scheduler.step()  # Moved to end of epoch

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with torch.amp.autocast('cuda', enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                baseline_loss = loss_fn(score, feat, target, target_cam)

                if dicma_module is not None and cfg.DICMA.ENABLED:
                    # Select feature tensor for DiCMA
                    dicma_feat = feat
                    if isinstance(feat, (list, tuple)):
                        if cfg.DICMA.USE_OVERLAPPING_PATCHES:
                            # Use patch features (index 3) for overlapping patches
                            dicma_feat = feat[3] if len(feat) > 3 and feat[3] is not None else feat[1]
                        else:
                            key = getattr(cfg.DICMA, 'FEAT_KEY', 1)
                            if isinstance(key, int) and 0 <= key < len(feat):
                                dicma_feat = feat[key]

                    # Prepare side information for DICMA
                    side_info = None
                    if cfg.DICMA.USE_SIDE_EMBEDDING:
                        side_info_list = []
                        if target_cam is not None:
                            side_info_list.append(target_cam.unsqueeze(-1).float())
                        if target_view is not None:
                            side_info_list.append(target_view.unsqueeze(-1).float())
                        if side_info_list:
                            side_info = torch.cat(side_info_list, dim=-1)

                    dicma_out = dicma_module(dicma_feat, target, side_info)
                    dicma_loss = cfg.DICMA.ALPHA * dicma_out.get('w2_loss', 0.0)
                    dicma_loss = dicma_loss + cfg.DICMA.BETA * dicma_out.get('cov_loss', 0.0)
                    dicma_loss = dicma_loss + cfg.DICMA.GAMMA * dicma_out.get('gw_loss', 0.0)
                    dicma_loss = dicma_loss + dicma_out.get('repulsion_loss', 0.0)

                    if cfg.DICMA.RETAIN_BASELINE:
                        loss = baseline_loss + dicma_loss
                    else:
                        loss = dicma_loss
                else:
                    loss = baseline_loss

            scaler.scale(loss).backward()

            # Add gradient clipping to prevent divergence
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                torch.nn.utils.clip_grad_norm_(center_criterion.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            if dicma_module is not None and cfg.DICMA.ENABLED:
                dicma_w2_meter.update(dicma_out.get('w2_loss', torch.tensor(0.)).item(), img.shape[0])
                dicma_cov_meter.update(dicma_out.get('cov_loss', torch.tensor(0.)).item(), img.shape[0])
                dicma_gw_meter.update(dicma_out.get('gw_loss', torch.tensor(0.)).item(), img.shape[0])
                dicma_rep_meter.update(dicma_out.get('repulsion_loss', torch.tensor(0.)).item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                log_msg = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                    epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]
                )
                if dicma_module is not None and cfg.DICMA.ENABLED:
                    log_msg += " | W2: {:.4f}, Cov: {:.4f}, GW: {:.4f}, Rep: {:.4f}".format(
                        dicma_w2_meter.avg, dicma_cov_meter.avg, dicma_gw_meter.avg, dicma_rep_meter.avg
                    )
                logger.info(log_msg)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # Step the scheduler at the end of each epoch
        scheduler.step()

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                           reranking=cfg.DICMA.USE_RERANK,
                           rerank_k1=cfg.DICMA.RERANK_K1,
                           rerank_k2=cfg.DICMA.RERANK_K2,
                           rerank_lambda=cfg.DICMA.RERANK_LAMBDA)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]