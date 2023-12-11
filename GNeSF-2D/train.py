import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
from loguru import logger

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tool.comm import is_main_process, synchronize
from gnesf.data_loaders import dataset_dict
from gnesf.render_ray import render_rays
from gnesf.render_image import render_single_image
from gnesf.model import IBRNetModel
from gnesf.sample_ray import RaySamplerSingleImage
from gnesf.criterion import Criterion, SemanticCriterion
from utils import mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, \
    create_log, celoss, calculate_segmentation_metrics, logits_2_label
import config
from gnesf.projection import Projector
from gnesf.data_loaders.create_training_dataset import create_training_dataset


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, args.expname)
    ckpt_dir = os.path.join(out_folder, 'ckpts')
    args.ckpt_dir = ckpt_dir
    if is_main_process():
        logger.info('outputs will be saved to {}'.format(out_folder))
        
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # save the args and config files
        f = os.path.join(out_folder, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

        if args.config is not None:
            f = os.path.join(out_folder, 'config.txt')
            if not os.path.isfile(f):
                shutil.copy(args.config, f)
        
        create_log(args, out_folder)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, 'val', # val validation
                                                  scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1) # , shuffle=False , shuffle=True
    val_loader_iterator = iter(cycle(val_loader))

    # Create IBRNet model
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    if args.enable_semantic: sem_criterion = SemanticCriterion(args.ignore_label) # .cuda()
    if is_main_process():
        writer = SummaryWriter(out_folder)
        logger.info('saving tensorboard files to {}'.format(out_folder))
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    # while global_step < model.start_step + args.n_iters + 1:
    while global_step < args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(1.0 * args.N_rand * args.num_source_views / train_data['src_rgbs'][0].shape[0])
            ray_batch = ray_sampler.random_sample(N_rand,
                                                  sample_mode=args.sample_mode,
                                                  center_ratio=args.center_ratio,
                                                  )
            
            src_rgbs = ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
            
            out = model.feature_net(src_rgbs) # *255 , src_sems
            if args.enable_2dsem_seg:
                featmaps, sem_pred2d = out
            else:
                sem_pred2d, featmaps = None, out[0]

            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              featmaps=featmaps,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd,
                              enable_rgb=args.enable_rgb,
                              enable_semantic=args.enable_semantic,
                              sem_pred2d=sem_pred2d)

            # compute loss
            model.optimizer.zero_grad()
            loss = 0.
            if args.enable_rgb:
                coarse_loss_rgb, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)
            
            if args.enable_semantic:
                coarse_loss_sem, scalars_to_log = sem_criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)

            if ret['outputs_fine'] is not None:
                if args.enable_rgb:
                    fine_loss_rgb, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                    loss_rgb = coarse_loss_rgb + fine_loss_rgb
                    loss += loss_rgb
                    
                if args.enable_semantic:
                    fine_loss_sem, scalars_to_log = sem_criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                    loss_sem = coarse_loss_sem + fine_loss_sem
                    loss += loss_sem * args.wgt_sem
                        

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if is_main_process():
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    if args.enable_rgb:
                        scalars_to_log['train/C-rgbL'] = coarse_loss_rgb.item()
                        scalars_to_log['train/C-psnr'] = mse2psnr(coarse_loss_rgb.item())
                    
                    if args.enable_semantic:
                        scalars_to_log['train/C-CeL'] = coarse_loss_sem
                    if ret['outputs_fine'] is not None:
                        if args.enable_rgb:
                            scalars_to_log['train/F-rgbL'] = fine_loss_rgb.item()
                            scalars_to_log['train/F-psnr'] = mse2psnr(fine_loss_rgb.item())
                            
                        if args.enable_semantic:
                            scalars_to_log['train/F-CeL'] = fine_loss_sem

                    logstr = 'Ep: {}  step: {} '.format(epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.1e}'.format(k.replace('train/', ''), scalars_to_log[k]) if scalars_to_log[k] < 0.01 else \
                            ' {}: {:.2f}'.format(k.replace('train/', ''), scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    logstr += ' iterT {:.02f} s'.format(dt)
                    logger.info(logstr)

                if global_step % args.i_weights == 0:
                    logger.info('Saving checkpoints at {} to {}...'.format(global_step, ckpt_dir))
                    # fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    fpath = os.path.join(ckpt_dir, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    logger.info('Logging a random validation view...')
                    val_data = next(val_loader_iterator)
                    tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3) if args.enable_rgb else None
                    gt_sem = tmp_ray_sampler.sem.reshape(H, W) if args.enable_semantic else None
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_sampler, projector,
                                   gt_img=gt_img, render_stride=args.render_stride, prefix='val/', gt_sem=gt_sem, 
                                   colour_map=val_dataset.colour_map)
                    torch.cuda.empty_cache()

                    logger.info('Logging current training view...')
                    tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device,
                                                                  render_stride=args.render_stride)
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3) if args.enable_rgb else None
                    gt_sem = tmp_ray_train_sampler.sem.reshape(H, W) if args.enable_semantic else None
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_train_sampler, projector,
                                   gt_img=gt_img, render_stride=args.render_stride, prefix='train/', gt_sem=gt_sem, 
                                   colour_map=train_dataset.colour_map)
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


def log_view_to_tb(writer, global_step, args, model, ray_sampler, projector, gt_img=None,
                   render_stride=1, prefix='', gt_sem=None, colour_map=None):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            src_rgbs = ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
            out = model.feature_net(src_rgbs)
            if args.enable_2dsem_seg:
                featmaps, sem_pred2d = out
            else:
                sem_pred2d, featmaps = None, out[0]
        else:
            featmaps = [None, None]
        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps,
                                  enable_rgb=args.enable_rgb,
                                  enable_semantic=args.enable_semantic,
                                  sem_pred2d=sem_pred2d)


    if gt_img is not None:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
        average_im = average_im[::render_stride, ::render_stride]
    if gt_sem is not None:
        gt_sem = gt_sem[::render_stride, ::render_stride]

    if gt_img is not None:
        average_im = img_HWC2CHW(average_im)

        rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

        rgb_gt = img_HWC2CHW(gt_img)
        h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
        w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
        rgb_im = torch.zeros(3, h_max, 3*w_max)
        rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
        rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
        rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred
    else:
        # ray_sampler.src_rgbs.cpu()
        rgb = ray_sampler.rgb.reshape(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride]
        # rgb_im = img_HWC2CHW(ray_sampler.src_rgbs.cpu().squeeze()[ray_sampler.src_rgbs.shape[1]//2])
        rgb_im = img_HWC2CHW(rgb)

    if gt_sem is not None:
        sem_pred = ret['outputs_coarse']['sem'].detach().cpu()
        sem_pred = logits_2_label(sem_pred.view(-1, sem_pred.shape[-1])) + 1
        h, w = gt_sem.shape[:2]
        
        sem_pred = sem_pred.view([h, w])
        sem_pred[gt_sem==args.ignore_label] = args.ignore_label + 1
        sem_pred = sem_pred.view(-1)
        
        sem_pred = img_HWC2CHW(colour_map[sem_pred].reshape(h, w, 3))
        sem_gt = img_HWC2CHW(colour_map[gt_sem.reshape(-1) + 1].reshape(h, w, 3))
        h_max = max(sem_gt.shape[-2], sem_pred.shape[-2])
        w_max = max(sem_gt.shape[-1], sem_pred.shape[-1])
        sem_im = torch.zeros(3, h_max, 2*w_max)
        sem_im[:, :sem_gt.shape[-2], :sem_gt.shape[-1]] = sem_gt
        sem_im[:, :sem_pred.shape[-2], w_max:w_max+sem_pred.shape[-1]] = sem_pred
        
    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        if gt_img is not None:
            rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
            rgb_fine_ = torch.zeros(3, h_max, w_max)
            rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
            rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
        if gt_sem is not None:
            sem_fine = ret['outputs_fine']['sem'].detach().cpu()
            sem_fine = logits_2_label(sem_fine) + 1
            
            sem_fine = sem_fine.view([h, w])
            sem_fine[gt_sem==args.ignore_label] = args.ignore_label + 1
            sem_fine = sem_fine.view(-1)
        
            sem_fine = img_HWC2CHW(colour_map[sem_fine].reshape(h_max, w_max, 3))
            
            sem_fine_ = torch.zeros(3, h_max, w_max)
            sem_fine_[:, :sem_fine.shape[-2], :sem_fine.shape[-1]] = sem_fine
            sem_im = torch.cat((sem_im, sem_fine_), dim=-1)
            
            writer.add_image(prefix + 'sem_gt-coarse-fine', sem_im, global_step)


    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    if gt_img is not None:
        # write scalar
        pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
        psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
        writer.add_scalar(prefix + 'psnr', psnr_curr_img, global_step)

    if gt_sem is not None:
        pred_sem = ret['outputs_fine']['sem'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['sem']
        ce_loss = celoss(pred_sem.reshape(-1, args.num_cls), gt_sem.reshape(-1), ignore_index=args.ignore_label).item()
        pred_sem = logits_2_label(pred_sem)
        
        miou, miou_valid_class, total_acc, avg_acc, ious = \
            calculate_segmentation_metrics(gt_sem.detach().cpu().numpy(), pred_sem.cpu().numpy(), args.num_cls, ignore_label=args.ignore_label)
        writer.add_scalar(prefix + 'mIoU', miou, global_step)
        writer.add_scalar(prefix + 'mIoU_validclass', miou_valid_class, global_step)
        writer.add_scalar(prefix + 'total_acc', total_acc, global_step)
        writer.add_scalar(prefix + 'avg_acc', avg_acc, global_step)
        writer.add_scalar(prefix + 'CeL', ce_loss, global_step)
    if True:
        logstr = ''
        if gt_img is not None:
            logstr += '{} {}psnr: {:.2f}'.format(args.expname, prefix, psnr_curr_img)
        if gt_sem is not None:
            logstr += ' CeL: {:.2f} mIoU: {:.2f} total_acc: {:.2f} avg_acc: {:.2f}'.format(ce_loss, miou, total_acc, avg_acc)
        logger.info(logstr)

    model.switch_to_train()


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)
