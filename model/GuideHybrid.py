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

                self.conv64_32_3 = nn.Conv2d(in_channels=up_features[0],
                                                out_channels=up_features[1],
                                                kernel_size=3,
                                                padding=1)
                self.conv64_1 = nn.Conv2d(in_channels=up_features[0],
                                        out_channels=up_features[0],
                                        kernel_size=1)
                self.conv64_3 = nn.Conv2d(in_channels=up_features[0],
                                        out_channels=up_features[0],
                                        kernel_size=3,
                                        padding=1)
                self.fp_conv128_64_1 = nn.Conv2d(in_channels=128,
                                        out_channels=up_features[0],
                                        kernel_size=1)
                self.conv128_64_3 = nn.Conv2d(in_channels=128,
                                        out_channels=up_features[0],
                                        kernel_size=3,
                                        padding=1)
                self.conv32_16_3 = nn.Conv2d(in_channels=up_features[1],
                                             out_channels=up_features[2],
                                             kernel_size=3,
                                             padding=1)
                
                # Special layers
                self.relu = nn.ReLU()

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
                                                out_features=1,
                                                kernel_size=3,
                                                channel_attention=True,
                                                guide_features=3,
                                                guidance_type="full")
                
                self.up_32 = Guided_Upsampling_Block(in_features=up_features[1],
                                                     expand_features=inner_features[1],
                                                     out_features=up_features[1],
                                                     channel_attention=True,
                                                     guide_features=3,
                                                     guidance_type='full')
                self.up_16 = Guided_Upsampling_Block(in_features=up_features[2],
                                                     expand_features=inner_features[2],
                                                     out_features=up_features[2],
                                                     channel_attention=True,
                                                     guide_features=3,
                                                     guidance_type='full')
                self.up_depth_final = Guided_Upsampling_Block(in_features=up_features[2],
                                                     expand_features=inner_features[2],
                                                     out_features=1,
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

                fp = self.relu(self.fp_conv128_64_1(fp)) # Reduced fp 128x60x80 --> 64x60x80

                # Concatenate the results on depth dimension
                concat_out = torch.cat((fp, x), dim=1) # 128 x 60 x 80
                support_map = x

                # Branch the results (depth)
                depth_feature = self.relu(self.conv128_64_3(concat_out)) # 64 x 60 x 80
                # depth_feature = self.relu(self.conv64_32_3(depth_feature)) # 32 x 60 x 80

                # Branch the results (segmentation)
                seg_feature = self.relu(self.conv128_64_3(concat_out)) # 64 x 60 x 80

                return seg_feature, support_map, depth_feature  # 64 x 60 x 80
        
        def seg_depth_module(self, seg_feature, support_map, depth_feature, img):
                """ Depth module to infere depth supported by a common representation and semantic feature """

                img_quarter = F.interpolate(img, scale_factor=0.25)
                img_half = F.interpolate(img, scale_factor=0.5)

                # Upsampling to quarter
                seg_feature = F.interpolate(seg_feature, scale_factor=2, mode='bilinear')
                support_map = F.interpolate(support_map, scale_factor=2, mode='bilinear')
                depth_feature = F.interpolate(depth_feature, scale_factor=2, mode='bilinear')

                # First upsample
                seg_feature = self.up_1(img_quarter, seg_feature) # 32 x 120 x 160
                support_map = self.up_1(img_quarter, support_map) # 32 x 120 x 160
                depth_feature = self.up_2(img_quarter, depth_feature)  # 16 x 120 x160

                seg_feature = F.interpolate(seg_feature, scale_factor=2, mode='bilinear')
                support_map = F.interpolate(support_map, scale_factor=2, mode='bilinear')
                depth_feature = F.interpolate(depth_feature, scale_factor=2, mode='bilinear')

                # Second upsample
                seg_feature = self.up_2(img_half, seg_feature) # 16 x 240 x 320
                support_map = self.up_2(img_half, support_map) # 16 x 240 x 320
                depth_feature = self.up_16(img_half, depth_feature)  # 16 x 240 x 320

                # Pixel multiplication
                combined_map = torch.mul(seg_feature, support_map) # 16 x 240 x 320
                combined_map = torch.cat((combined_map, depth_feature), dim=1) # 32 x 240 x 320

                # Final step
                combined_map = self.relu(self.conv32_16_3(combined_map)) # 16 x 240 x 320

                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                depth = self.up_depth_final(img, combined_map)

                return depth
        
        def dept_seg_module(self, seg_feature, support_map, depth_feature, img):

                img_quarter = F.interpolate(img, scale_factor=0.25)
                img_half = F.interpolate(img, scale_factor=0.5)

                # Upsampling to quarter
                seg_feature = F.interpolate(seg_feature, scale_factor=2, mode='bilinear')
                support_map = F.interpolate(support_map, scale_factor=2, mode='bilinear')
                depth_feature = F.interpolate(depth_feature, scale_factor=2, mode='bilinear')

                # First upsample
                seg_feature = self.up_1(img_quarter, seg_feature) # 32 x 120 x 160
                support_map = self.up_1(img_quarter, support_map) # 32 x 120 x 160
                depth_feature = self.up_32(img_quarter, depth_feature)  # 32 x 120 x160

                seg_feature = F.interpolate(seg_feature, scale_factor=2, mode='bilinear')
                support_map = F.interpolate(support_map, scale_factor=2, mode='bilinear')
                depth_feature = F.interpolate(depth_feature, scale_factor=2, mode='bilinear')

                # Second upsample
                seg_feature = self.up_2(img_half, seg_feature) # 16 x 240 x 320
                support_map = self.up_2(img_half, support_map) # 16 x 240 x 320
                depth_feature = self.up_2(img_half, depth_feature)  # 16 x 240 x 320

                # Pixel mutiplication and square root
                combined_map = torch.mul(support_map, depth_feature) # 16 x 240 x 320
                combined_map = torch.cat((seg_feature, combined_map), dim=1) # 32 x 240 x 320

                # Final step
                combined_map = self.relu(self.conv32_16_3(combined_map)) # 16 x 240 x 320

                combined_map = F.interpolate(combined_map, scale_factor=2, mode='bilinear')
                segmentation = self.up_seg_final(img, combined_map)

                return segmentation


        def forward(self, x):
                feature_map, fp = self.feature_extractor(x)

                seg_feature, support_map, depth_feature = self.decoder(feature_map, fp)

                depth = self.seg_depth_module(seg_feature, support_map, depth_feature, x)

                segmentation = self.dept_seg_module(seg_feature, support_map, depth_feature, x)

                return depth, segmentation