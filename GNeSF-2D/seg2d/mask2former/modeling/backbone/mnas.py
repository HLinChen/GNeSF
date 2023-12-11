
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0):
        super(MnasMulti, self).__init__()
        depths = _get_depths(alpha) # channels mobilenet
        if alpha == 1.0:
            # MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
            MNASNet = torchvision.models.mnasnet1_0(weights='MNASNet1_0_Weights.IMAGENET1K_V1', progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            MNASNet.layers._modules['8'],
        )

        self.conv1 = MNASNet.layers._modules['9']
        self.conv2 = MNASNet.layers._modules['10']
        self.conv3 = nn.Sequential(
            MNASNet.layers._modules['11'],
            MNASNet.layers._modules['12'],
        )
        
        
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": depths[2],
            "res3": depths[3],
            "res4": depths[4],
            "res5": depths[6],
        }

        # self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False) # 80 80
        # self.out_channels = [depths[4]] # 80

        # final_chs = depths[4] # 80
        # self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True) # 40 80
        # self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True) # 24 80

        # self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False) # 80 40
        # self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False) # 80 24
        # self.out_channels.append(depths[3])
        # self.out_channels.append(depths[2]) # [80, 40, 24]

    def forward(self, x): # bs, 3, 480, 640
        # print(x.shape)
        # # exit()
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        
        outs = {
            "res2": conv0,  # bs, 24, 120, 160 H // 4
            "res3": conv1,  # bs, 40, 60, 80 H // 8
            "res4": conv2,  # bs, 80, 30, 40 H // 16
            "res5": conv3,  # bs, 192, 15, 20 H // 32
        }

        # # FPN
        # intra_feat = conv2
        # outputs = []
        # out = self.out1(intra_feat)
        # outputs.append(out) # bs, 80, 30, 40 H // 16

        # intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        # out = self.out2(intra_feat)
        # outputs.append(out) # bs, 40, 60, 80 H // 8

        # intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        # out = self.out3(intra_feat)
        # outputs.append(out) # bs, 24, 120, 160 H // 4

        # return outputs[::-1] # scale: big[0] -> small[1]
        return outs # scale: big[0] -> small[1]


@BACKBONE_REGISTRY.register()
class D2MnasMulti(MnasMulti, Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(alpha=cfg.MODEL.MNAS.ALPHA)
        
        self._out_features = cfg.MODEL.MNAS.OUT_FEATURES
        
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    # @property
    # def size_divisibility(self):
    #     return 32