import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DDRNet_23_slim import DualResNet_Backbone
from model.modules import Guided_Upsampling_Block

class GuideHybrid(nn.Module):
        def __init__(self, 
                        pretrained=True,
                        up_features=[64, 32, 16], 
                        inner_features=[64, 32, 16]):
                super(GuideHybrid, self).__init__()

                self.feature_extractor = DualResNet_Backbone(
                        pretrained=pretrained, 
                        features=up_features[0])

                self.conv64_1 = nn.Conv2d(in_channels=64,
                                        out_channels=64,
                                        kernel_size=1)
                self.conv32_1 = nn.Conv2d(in_channels=32,
                                          out_channels=32,
                                          kernel_size=1)
                self.fp_conv128_64_1 = nn.Conv2d(in_channels=128,
                                        out_channels=up_features[0],
                                        kernel_size=1)
                self.conv128_64_3 = nn.Conv2d(in_channels=128,
                                        out_channels=up_features[0],
                                        kernel_size=3,
                                        padding=1)
                self.seg_depth_preconv = nn.Sequential(
                        nn.Conv2d(in_channels=64,
                                        out_channels=2,
                                        kernel_size=3,
                                        padding=1),
                        nn.BatchNorm2d(2),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=2,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                )
                self.depth_seg_preconv = nn.Sequential(
                        nn.Conv2d(in_channels=64,
                                        out_channels=64,
                                        kernel_size=3,
                                        padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64,
                                        out_channels=32,
                                        kernel_size=3,
                                        padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                )
                
                # Special layers
                self.relu = nn.ReLU()

                # Guided Upsampling for Depth
                self.up_depth_1 = Guided_Upsampling_Block(in_features=up_features[0]+1,
                                                        expand_features=inner_features[0]+1,
                                                        out_features=up_features[1],
                                                        kernel_size=3,
                                                        channel_attention=True,
                                                        guide_features=3,
                                                        guidance_type='full')
                self.up_depth_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                                        expand_features=inner_features[1],
                                                        out_features=up_features[2],
                                                        kernel_size=3,
                                                        channel_attention=True,
                                                        guide_features=3,
                                                        guidance_type='full')
                self.up_depth_final = Guided_Upsampling_Block(in_features=up_features[2],
                                                     expand_features=inner_features[2],
                                                     out_features=1,
                                                     channel_attention=True,
                                                     guide_features=3,
                                                     guidance_type='full')
                

                # Guided Upsampling for Segmentation
                self.up_seg_1 =Guided_Upsampling_Block(in_features=up_features[0]+32,
                                                        expand_features=inner_features[0]+32,
                                                        out_features=up_features[1],
                                                        kernel_size=3,
                                                        channel_attention=True,
                                                        guide_features=3,
                                                        guidance_type='full')
                self.up_seg_2 =Guided_Upsampling_Block(in_features=up_features[1],
                                                        expand_features=inner_features[1],
                                                        out_features=up_features[2],
                                                        kernel_size=3,
                                                        channel_attention=True,
                                                        guide_features=3,
                                                        guidance_type='full')
                self.up_seg_final = Guided_Upsampling_Block(in_features=up_features[2],
                                                     expand_features=inner_features[2],
                                                     out_features=14,
                                                     channel_attention=True,
                                                     guide_features=3,
                                                     guidance_type='full')
                

        def decoder(self, x, fp):

                x = self.relu(self.conv64_1(x)) # 64 x 60 x 80

                # fp = self.relu(self.fp_conv128_64_1(fp)) # Reduced fp 128x60x80 --> 64x60x80

                # Concatenate the results on depth dimension
                concat_out = torch.cat((fp, x), dim=1) # 128 x 60 x 80
                support_map = x

                # Branch the results (depth)
                depth_feature = self.relu(self.conv128_64_3(concat_out)) # 64 x 60 x 80

                # Branch the results (segmentation)
                seg_feature = self.relu(self.conv128_64_3(concat_out)) # 64 x 60 x 80

                return seg_feature, support_map, depth_feature  # 64 x 60 x 80
        
        def seg_depth_module(self, seg_feature, support_map, depth_feature, img):
                """ Depth module to infere depth supported by a common representation and semantic feature """

                img_quarter = F.interpolate(img, scale_factor=0.25)
                img_half = F.interpolate(img, scale_factor=0.5)

                seg_feature = self.seg_depth_preconv(seg_feature) # 1 x 60 x 80
                support_map = self.seg_depth_preconv(support_map) # 1 x 60 x 80

                # Pixel multiplication
                combined_map = torch.mul(seg_feature, support_map) # 1 x 60 x 80
                combined_map = torch.cat((combined_map, depth_feature), dim=1) # 65 x 60 x 80

                # Final step
                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                combined_map = self.up_depth_1(img_quarter, combined_map) # 16 x 120 x 160

                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                combined_map = self.up_depth_2(img_half, combined_map) # 8 x 240 x 320

                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                depth = self.up_depth_final(img, combined_map) # 1 x 480 x 640

                return depth
        
        def dept_seg_module(self, seg_feature, support_map, depth_feature, img):

                img_quarter = F.interpolate(img, scale_factor=0.25)
                img_half = F.interpolate(img, scale_factor=0.5)

                depth_feature = self.depth_seg_preconv(depth_feature)
                support_map = self.depth_seg_preconv(support_map)

                # Pixel mutiplication
                combined_map = torch.mul(support_map, depth_feature) # 32 x 60 x 80
                combined_map = self.conv32_1(combined_map) # 32 x 60 x 80
                combined_map = torch.cat((seg_feature, combined_map), dim=1) # 96 x 60 x 80

                # Final step
                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                combined_map = self.up_seg_1(img_quarter, combined_map) # 32 x 120 x 160

                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                combined_map = self.up_seg_2(img_half, combined_map) # 16 x 240 x 320

                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                segmentation = self.up_seg_final(img, combined_map) # 14 x 480 x 640

                return segmentation


        def forward(self, x):
                feature_map, fp = self.feature_extractor(x)

                seg_feature, support_map, depth_feature = self.decoder(feature_map, fp)

                depth = self.seg_depth_module(seg_feature, support_map, depth_feature, x)
                segmentation = self.dept_seg_module(seg_feature, support_map, depth_feature, x)

                return depth, segmentation