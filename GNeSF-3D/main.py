import argparse
import os
import time
import datetime
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger
from tqdm import tqdm

from utils import tensor2float, save_scalars, DictAverageMeter, SaveScene, make_nograd_func, ImageWriter, info_print #, compute_psnr
from datasets import transforms, find_dataset_def
from models import NeuralRecon
from config import cfg, update_config
from datasets.sampler import DistributedSampler
from ops.comm import *


def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of NeuralRecon')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./config/train.yaml',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='env://', # tcp://127.0.0.1:23456
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    # parse arguments and check
    args = parser.parse_args()

    return args


def configure_optimizers(model):
    # allow for different learning rates between pretrained layers 
    # (resnet backbone) and new layers (everything else).
    params_backbone2d = model.backbone2d.parameters()
    params_backbone3d = model.neucon_net.sp_convs.parameters()
    if cfg.MODEL.FUSION.FUSION_ON:
        params_fusion = model.neucon_net.gru_fusion.parameters()
    params_heads3d_sdf = model.neucon_net.tsdf_preds.parameters()
    params_heads3d_occ = model.neucon_net.occ_preds.parameters()
    
    if cfg.MODEL.ENABLE_RENDER:
        modules_nerf = [model.neucon_net.render_network.render.mlp] # model.render.ln_beta
        params_nerf = list(*(p.parameters() for p in modules_nerf))
        modules_nerf = [model.neucon_net.render_network.fuse_fpn] # model.render.ln_beta
        params_nerf += list(*(p.parameters() for p in modules_nerf))
        params_nerf += [model.neucon_net.render_network.render.ln_beta]
        
    
    params = []
    # optimzer
    lr = cfg.TRAIN.LR
    lr_backbone2d = lr
    
    if not cfg.MODEL.FREEZE_2D:
        params.append({'params': params_backbone2d, 'lr': lr_backbone2d})
    else:
        for i in model.backbone2d.parameters(): i.requires_grad = False
        
    if not cfg.MODEL.FREEZE_3DBACKBONE:
        params.append({'params': params_backbone3d, 'lr': lr})
        if cfg.MODEL.FUSION.FUSION_ON:
            params.append({'params': params_fusion, 'lr': lr})
    else:
        for i in model.neucon_net.sp_convs.parameters(): i.requires_grad = False
        if cfg.MODEL.FUSION.FUSION_ON:
            for i in model.neucon_net.gru_fusion.parameters(): i.requires_grad = False
        
    if not cfg.MODEL.FREEZE_3DHEAD:
        params.append({'params': params_heads3d_sdf, 'lr': lr})
        params.append({'params': params_heads3d_occ, 'lr': lr})
    else:
        for i in model.neucon_net.tsdf_preds.parameters(): i.requires_grad = False
        for i in model.neucon_net.occ_preds.parameters(): i.requires_grad = False
    
    if cfg.MODEL.ENABLE_RENDER:
        lr_nerf = lr
        params.append({'params': params_nerf, 'lr': lr_nerf})
            
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD) # , lr=cfg.TRAIN.LR
    
    return optimizer


def modefy_name_state_dict(state_dict):
    for k in list(state_dict.keys()):
        newk = k
        if "module.backbone2d" in k and not k.startswith('module.backbone2d.backbone2d'):
            newk = k.replace('module.backbone2d', 'module.backbone2d.backbone2d')
            # logger.debug(f"{k} ==> {newk}")
        if newk != k:
            state_dict[newk] = state_dict[k]
            del state_dict[k]
            
    return state_dict


args = args()
update_config(cfg, args)

cfg.defrost()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

cfg.DISTRIBUTED = num_gpus > 1

if cfg.DISTRIBUTED:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl"
    )
    synchronize()
cfg.LOCAL_RANK = args.local_rank
cfg.freeze()

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# create logger
if is_main_process():
    if not os.path.isdir(cfg.LOGDIR):
        os.makedirs(cfg.LOGDIR)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, f'{current_time_str}_{cfg.MODE}.log')
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO", enqueue=True)

    tb_writer = SummaryWriter(cfg.LOGDIR)
    
    print('number of gpus: {}'.format(num_gpus))
    logger.info('\n{}'.format(cfg))

# Augmentation
if cfg.MODE == 'train':
    n_views = cfg.TRAIN.N_VIEWS
    random_rotation = cfg.TRAIN.RANDOM_ROTATION_3D
    random_translation = cfg.TRAIN.RANDOM_TRANSLATION_3D
    paddingXY = cfg.TRAIN.PAD_XY_3D
    paddingZ = cfg.TRAIN.PAD_Z_3D
else:
    n_views = cfg.TEST.N_VIEWS
    random_rotation = False
    random_translation = False
    paddingXY = 0
    paddingZ = 0

transform = []
transform += [transforms.ResizeImage((cfg.WIDTH, cfg.HEIGHT), ignore_label=cfg.IGNORE_LABEL),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
              transforms.IntrinsicsPoseToProjection(n_views, 4),
              ]
transform_test = [transforms.ResizeImage((cfg.WIDTH, cfg.HEIGHT), ignore_label=cfg.IGNORE_LABEL),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, False, False,
                  0, 0, max_epoch=cfg.TRAIN.EPOCHS),
              transforms.IntrinsicsPoseToProjection(n_views, 4),
              ]

transforms_test = transforms.Compose(transform_test)
transforms = transforms.Compose(transform)

# dataset, dataloader
MVSDataset = find_dataset_def(cfg.DATASET)
train_dataset = MVSDataset(cfg.TRAIN.PATH, "train", transforms, cfg.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, cfg)
if cfg.MODE == 'train':
    test_dataset = MVSDataset(cfg.TEST.PATH, "val", transforms_test, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, cfg)
else:
    test_dataset = MVSDataset(cfg.TEST.PATH, "val", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, cfg)

if cfg.DISTRIBUTED:
    train_sampler = DistributedSampler(train_dataset, shuffle=False)
    TrainImgLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=cfg.TRAIN.N_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    TestImgLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=test_sampler,
        num_workers=cfg.TEST.N_WORKERS,
        pin_memory=True,
        drop_last=False
    )
else:
    TrainImgLoader = DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.N_WORKERS,
                                drop_last=True)
    TestImgLoader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS,
                               drop_last=False)

# model, optimizer
model = NeuralRecon(cfg)
optimizer = configure_optimizers(model)
if cfg.DISTRIBUTED:
    model.cuda()
    model = DistributedDataParallel(
        model, device_ids=[cfg.LOCAL_RANK], output_device=cfg.LOCAL_RANK,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
        find_unused_parameters=True
    )
else:
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()

img_val_writer = ImageWriter(cfg.LOGDIR, mode='val')

# main function
def train():
    # load parameters
    start_epoch = 0
    if cfg.RESUME:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(saved_models) != 0:
            # use the latest checkpoint file
            loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
            if is_main_process(): logger.info("resuming " + str(loadckpt))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
            state_dict = torch.load(loadckpt, map_location=map_location)
            model.load_state_dict(state_dict['model'], strict=False)
            lr = state_dict['optimizer']['param_groups'][0]['lr']
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['initial_lr'] = lr
                optimizer.param_groups[i]['lr'] = lr
            start_epoch = state_dict['epoch'] + 1
    elif cfg.LOADCKPT != '':
        # load checkpoint file specified by args.loadckpt
        if is_main_process(): logger.info("loading model {}".format(cfg.LOADCKPT))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
        state_dict = torch.load(cfg.LOADCKPT, map_location=map_location)
        state_dict['model'] = modefy_name_state_dict(state_dict['model'])
        model.load_state_dict(state_dict['model'], strict=False)
        lr = state_dict['optimizer']['param_groups'][0]['lr']
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['initial_lr'] = lr
            optimizer.param_groups[i]['lr'] = lr
    elif cfg.PRETRAIN != '':
        # load checkpoint file specified by args.loadckpt
        if is_main_process(): logger.info("loading model {}".format(cfg.PRETRAIN))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
        state_dict = torch.load(cfg.PRETRAIN, map_location=map_location)
        state_dict['model'] = modefy_name_state_dict(state_dict['model'])
        model.load_state_dict(state_dict['model'], strict=False)
        if cfg.MODEL.SEMSEG2D.PRETRAIN_PATH: model.module.backbone2d.load_sem_seg(cfg.MODEL.SEMSEG2D.PRETRAIN_PATH)
    if is_main_process():
        logger.info("start at epoch {}".format(start_epoch))
        logger.info('Number of model parameters: {}M'.format(sum([p.data.nelement() for p in model.parameters()])/1e6))
        avg_train_scalars = DictAverageMeter()

    milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
    lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, cfg.TRAIN.EPOCHS):
        lr_scheduler.step()
        if is_main_process(): logger.info('Epoch {} LR {}:'.format(epoch_idx, lr_scheduler.get_last_lr()[0]))
        TrainImgLoader.dataset.epoch = epoch_idx
        TrainImgLoader.dataset.tsdf_cashe = {}
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % cfg.SUMMARY_FREQ == 0
            start_time = time.time()
            loss, scalar_outputs, outputs = train_sample(sample)
            if (batch_idx + 1) % cfg.PRINT_FREQ == 0 and is_main_process():
                info = 'Ep {}/{}, Iter {}/{}, loss: {:.3f}, time: {:.2f}'.format(epoch_idx, cfg.TRAIN.EPOCHS,
                                                                                         batch_idx,
                                                                                         len(TrainImgLoader), loss,
                                                                                         time.time() - start_time)
                if cfg.MODEL.ENABLE_RENDER and 'psnr' in scalar_outputs:
                    info += ' psnr: {:.2f}'.format(scalar_outputs['psnr'])
                logger.info(info)
            if is_main_process(): avg_train_scalars.update(scalar_outputs)
            
            if do_summary and is_main_process():
                save_scalars(tb_writer, 'train', scalar_outputs, global_step)
            del scalar_outputs, outputs
        
        if is_main_process():
            avg_scalars = avg_train_scalars.mean()
            save_scalars(tb_writer, 'fulltrain', avg_scalars, epoch_idx)
            info_print(epoch_idx, avg_scalars, mode='Train')
        

        # checkpoint
        if (epoch_idx + 1) % cfg.SAVE_FREQ == 0 and is_main_process():
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(cfg.LOGDIR, epoch_idx))
            
        # lr_scheduler.step()
        if cfg.MODEL.ENABLE_RENDER and (epoch_idx + 1) % cfg.VAL_FREQ == 0:
            val(epoch_idx)
        
        if not cfg.MODEL.FUSION.FUSION_ON and (epoch_idx + 1) >= cfg.TRAIN.EPOCH_PHASE1:
            if is_main_process(): logger.info("Finish phase 1!")
            return
    val(epoch_idx)


def val(epoch_idx):
    with torch.no_grad():
        avg_val_scalars = DictAverageMeter()
        TestImgLoader.dataset.epoch = epoch_idx
        TestImgLoader.dataset.tsdf_cashe = {}
        save_mesh_scene = SaveScene(cfg)
        batch_len = len(TestImgLoader)
        
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % cfg.SUMMARY_FREQ == 0
            start_time = time.time()
            save_scene = cfg.SAVE_SCENE_MESH and (batch_idx == batch_len - 1)
            loss, scalar_outputs, outputs = test_sample(sample, save_scene)
            if (batch_idx + 1) % cfg.PRINT_FREQ == 0 and is_main_process():
                info = 'Ep {}/{}, Iter {}/{}, loss: {:.3f}, time: {:.2f}'.format(epoch_idx, cfg.TRAIN.EPOCHS,
                                                                                         batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                         time.time() - start_time)
                if cfg.MODEL.ENABLE_RENDER:
                    info += ' psnr: {:.2f}'.format(scalar_outputs['psnr'])
                logger.info(info)
                
            # save mesh
            if cfg.SAVE_SCENE_MESH:
                # print('Saving...')
                save_mesh_scene(outputs, sample, epoch_idx)
                
            avg_val_scalars.update(scalar_outputs)
            del scalar_outputs
            
            if cfg.MODEL.ENABLE_RENDER and (batch_idx + 1) % cfg.VIS_FREQ == 0 and len(outputs) > 0:
            # if cfg.MODEL.ENABLE_RENDER:
                names = ['{}_val_{}_{}'.format(scene, batch_idx, get_rank()) for scene in sample['scene']]
                img_val_writer.save_image(sample, outputs, names, render_stride=cfg.MODEL.NERF.SAMPLE.RENDER_STRIDE)
        
        if is_main_process():
            avg_scalars = avg_val_scalars.mean()
            save_scalars(tb_writer, 'fullval', avg_scalars, epoch_idx)
            info_print(epoch_idx, avg_scalars, mode='Val')
        
    return


def test(from_last_num=0):
    ckpt_list = []
    result_dic = {}
    while True:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        if from_last_num != 0:
            saved_models = saved_models[-from_last_num:]
        for ckpt in saved_models:
            if ckpt not in ckpt_list: #  and int(ckpt.split('.')[0][-2:]) >= 20
                # use the latest checkpoint file
                loadckpt = os.path.join(cfg.LOGDIR, ckpt)
                if is_main_process(): logger.info("resuming " + str(loadckpt))
                state_dict = torch.load(loadckpt, map_location='cpu')
                model.load_state_dict(state_dict['model'])
                epoch_idx = state_dict['epoch']

                TestImgLoader.dataset.tsdf_cashe = {}

                avg_test_scalars = DictAverageMeter()
                save_mesh_scene = SaveScene(cfg)
                batch_len = len(TestImgLoader)
                for batch_idx, sample in enumerate(TestImgLoader):

                    # save mesh if SAVE_SCENE_MESH and is the last fragment
                    save_scene = cfg.SAVE_SCENE_MESH and (batch_idx == batch_len - 1)

                    start_time = time.time()
                    loss, scalar_outputs, outputs = test_sample(sample, save_scene)
                    if is_main_process() and batch_idx % 1000 == 0:
                        logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                                    len(TestImgLoader),
                                                                                                    loss,
                                                                                                    time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs

                    # save mesh
                    if cfg.SAVE_SCENE_MESH:
                        # print('Saving...')
                        save_mesh_scene(outputs, sample, epoch_idx)
                if is_main_process():
                    save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), epoch_idx)
                    logger.info("epoch {} avg_test_scalars: {}".format(epoch_idx, avg_test_scalars.mean()))
                    result_dic[ckpt] = avg_test_scalars.mean()


                ckpt_list.append(ckpt)

        return


def train_sample(sample):
    model.train()
    optimizer.zero_grad()

    outputs, loss_dict = model(sample)
    loss = loss_dict['total_loss']
    if not (len(outputs) == 0 and cfg.MODEL.FREEZE_2D and cfg.MODEL.FREEZE_3DBACKBONE and cfg.MODEL.FREEZE_3DHEAD):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return tensor2float(loss), tensor2float(loss_dict), outputs


@make_nograd_func
def test_sample(sample, save_scene=False):
    model.eval()

    outputs, loss_dict = model(sample, save_scene)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs


if __name__ == '__main__':
    if cfg.MODE == "train":
        train()
    elif cfg.MODE == "test":
        test(from_last_num=1)
