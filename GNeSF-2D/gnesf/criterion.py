import torch.nn as nn
from utils import img2mse, celoss


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask'].float()
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log


class SemanticCriterion(nn.Module):
    def __init__(self, ignore_label=0):
        super().__init__()
        self.ignore_label = ignore_label
        

    def forward(self, outputs, ray_batch, scalars_to_log):
        '''
        training semantic criterion
        '''
        pred_sem = outputs['sem']
        pred_mask = outputs['mask'].bool()
        gt_sem = ray_batch['sem']
        gt_sem = gt_sem.masked_fill(pred_mask == False, self.ignore_label)
        
        loss = celoss(pred_sem, gt_sem, ignore_index=self.ignore_label)

        return loss, scalars_to_log

