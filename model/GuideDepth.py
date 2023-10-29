import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DDRNet_23_slim import DualResNet_Backbone
from model.decoders import UpSamplingDecoder


class GuideDepth(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(GuideDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.depth_decoder = UpSamplingDecoder(inner_features=inner_features,
                                               up_features=up_features,
                                               task='depth')
        
        self.seg_decoder = UpSamplingDecoder(inner_features=inner_features,
                                               up_features=up_features,
                                               task='segmentation')


    # def _decoder(self, x, features):
    #     x_half = F.interpolate(x, scale_factor=.5)
    #     x_quarter = F.interpolate(x, scale_factor=.25)

    #     y = F.interpolate(features, scale_factor=2, mode='bilinear')
    #     y = self.up_1(x_quarter, y)

    #     y = F.interpolate(y, scale_factor=2, mode='bilinear')
    #     y = self.up_2(x_half, y)

    #     y = F.interpolate(y, scale_factor=2, mode='bilinear')

    #     return y


    def forward(self, x):
        y = self.feature_extractor(x)

        depth = self.depth_decoder(x, y)
        segment = self.seg_decoder(x, y)

        # # Upsample block for depth estimation
        # prev_depth = self._decoder(x, y)
        # depth = self.up_3(x, prev_depth)


        # # Upsample block for segementaion
        # prev_segment = self._decoder(x, y)
        # segment = self.up_3_seg(x, prev_segment)
        
        return depth, segment
