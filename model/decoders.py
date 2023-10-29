import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import Guided_Upsampling_Block


class UpSamplingDecoder(nn.Module):
    def __init__(self, inner_features:list, up_features:list, task:str):
        super(UpSamplingDecoder, self).__init__()

        if task not in ['depth', 'segmentation']:
            print('\n[ERROR] Task selected for decoder not valid. Check spelling for: depth or segmentation\n')
            exit(0)

        self.out_features = 1 if task == 'depth' else 14

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=self.out_features,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
    
    def forward(self, x, features):
        """
            Description:

            Inputs:
                x <Tensor>: original image before encoder
                features <Tensor>: features extracted from the encoder

            Returns:
                y <Tensor>: output of the decoder
        """
        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)

        y = F.interpolate(features, scale_factor=2, mode='bilinear')
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_3(x, y)

        return y