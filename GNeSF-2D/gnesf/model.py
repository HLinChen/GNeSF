# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import os
from loguru import logger
from detectron2.projects.deeplab.lr_scheduler import WarmupPolyLR

from gnesf.mlp_network import IBRNet
from gnesf.backbone import Encoderdecoder2D
from tool.comm import is_main_process


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class IBRNetModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device('cuda:{}'.format(args.local_rank))
        # create coarse IBRNet
        self.net_coarse = IBRNet(args,
                                 in_feat_ch=self.args.coarse_feat_dim,
                                 n_samples=self.args.N_samples).to(device)
        
        if args.coarse_only:
            self.net_fine = None
        else:
            self.net_fine = IBRNet(args,
                                   in_feat_ch=self.args.fine_feat_dim,
                                   n_samples=self.args.N_samples+self.args.N_importance).to(device)

        self.feature_net = Encoderdecoder2D(args).cuda()
        learnable_params = list(self.feature_net.feature_net.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam([
                {'params': self.net_coarse.parameters()},
                {'params': self.net_fine.parameters()},
                {'params': learnable_params, 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.net_coarse.parameters()},
                {'params': learnable_params, 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)

        self.scheduler = WarmupPolyLR(self.optimizer, max_iters=args.n_iters+1)

        out_folder = os.path.join(args.rootdir, args.expname)
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'net_coarse': de_parallel(self.net_coarse).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict()
                   }

        if self.net_fine is not None:
            to_save['net_fine'] = de_parallel(self.net_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.net_coarse.load_state_dict(to_load['net_coarse'])
        
        self.feature_net.load_state_dict(to_load['feature_net'])

        if self.net_fine is not None and 'net_fine' in to_load.keys():
            self.net_fine.load_state_dict(to_load['net_fine'])

    def modify_state(self, old_state, key='rgb_fc', new_key='weight_fc'):
        new_state = {}
        
        for k, v in old_state.items():
            if k.startswith(key):
                print('change {} to {}'.format(k, k.replace(key, 'rgb_fc')))
                new_state[k.replace(key, new_key)] = v
            else:
                new_state[k] = v
        
        return new_state

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        ckpts = []
        ckpt_dir = os.path.join(out_folder, 'ckpts')
        if os.path.exists(ckpt_dir):
            ckpts = [os.path.join(ckpt_dir, f)
                     for f in sorted(os.listdir(ckpt_dir)) if f.endswith('.pth')]
        
        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            if is_main_process(): logger.info('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            if is_main_process(): logger.info('No ckpts found, training from scratch...')
            step = 0

        return step
