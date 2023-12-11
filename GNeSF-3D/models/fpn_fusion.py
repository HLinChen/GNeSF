import torch
import torch.nn as nn
import torch.nn.functional as F



class FPNFusion(nn.Module):
    def __init__(self, out_ch_2d, mode='cat') -> None:
        super().__init__()
        self.mode = mode
        
        if mode == 'cat':
            self.smooth0 = nn.Conv2d(out_ch_2d[0], 8, kernel_size=3, stride=1, padding=1)
            self.smooth1 = nn.Conv2d(out_ch_2d[1], 16, kernel_size=3, stride=1, padding=1)
            self.smooth2 = nn.Conv2d(out_ch_2d[2], 32, kernel_size=3, stride=1, padding=1)
    
    def forward(self, feats, sem_seg=None):
        """
        get feature maps of all feats
        :param feats: list: [[scale, scale, scale], ...] scale: [bs, c, h, w]
        :return:
        """
        fused_feature_maps = []
        
        for pyramid_feat in feats: # each view; pyramid_feat: [bs, c, h, w]
            if self.mode == 'cat':
                # * the pyramid features are very important, if only use the coarst features, hard to optimize
                fused_feature_map = torch.cat([
                    F.interpolate(self.smooth2(pyramid_feat[2]), scale_factor=4, mode='bilinear'), # , align_corners=True
                    F.interpolate(self.smooth1(pyramid_feat[1]), scale_factor=2, mode='bilinear'), # , align_corners=True
                    self.smooth0(pyramid_feat[0])
                ], dim=1)
            elif self.mode == 'last':
                fused_feature_map = pyramid_feat[0]
            else:
                raise NotImplementedError
            
            fused_feature_map = F.interpolate(fused_feature_map, scale_factor=4, mode='bilinear')
            fused_feature_maps.append(fused_feature_map)
        
        fused_feature_maps = torch.stack(fused_feature_maps, dim=1) # [bs, views, c, h, w]
        
        if sem_seg is not None:
            fused_feature_maps = torch.cat([sem_seg, fused_feature_maps], dim=2)

        return fused_feature_maps # [v, 56, h, w]
        