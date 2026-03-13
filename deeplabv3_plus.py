import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models as models

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module for DeepLabV3+.
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different rates
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution after concatenation
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.shape[2:]
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        
        global_avg = self.global_avg_pool(x)
        global_avg = F.interpolate(global_avg, size=size, mode='bilinear', align_corners=True)
        
        out = torch.cat([conv1, conv2, conv3, conv4, global_avg], dim=1)
        out = self.conv_out(out)
        out = self.dropout(out)
        
        return out

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model for water segmentation.
    Modified to support 6-channel input (Sentinel-2 bands).
    """
    def __init__(self, n_channels=6, n_classes=1, output_stride=16, pretrained_backbone=True, backbone_type='resnet50'):
        super(DeepLabV3Plus, self).__init__()
        
        # Input validation
        if n_channels != 6:
            raise ValueError(f"Expected 6 input channels for Sentinel-2 data, got {n_channels}")
        
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
            aspp_out_channels = 256
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
            aspp_out_channels = 256
        else:
            raise ValueError('Output stride must be 8 or 16.')
        
        # Store backbone type for later use
        self.backbone_type = backbone_type
        
        # Backbone (using ResNet-50 or MobileNetV2 modified for 6-channel input)
        self.backbone = self._make_backbone(n_channels, pretrained_backbone, output_stride, backbone_type)
        
        # ASPP module (will be updated in _make_backbone for MobileNetV2)
        if backbone_type != 'mobilenet_v2':
            self.aspp = ASPP(2048, aspp_out_channels, atrous_rates)
        
        # Decoder (will be updated in _make_backbone for MobileNetV2)
        if backbone_type != 'mobilenet_v2':
            # Add a 1x1 convolution to reduce low-level features channels from 1024 to 256
            # This is necessary because low_level_features from ResNet-50 layer3 has 1024 channels,
            # but we need to concatenate it with ASPP output (256 channels) to get 512 channels
            # which is the expected input for the first convolution in the decoder
            self.reduce_low_level = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()

    def _make_backbone(self, n_channels, pretrained, output_stride, backbone_type='resnet50'):
        """
        Create a modified backbone for 6-channel input.
        Supports both ResNet-50 and MobileNetV2.
        """
        if backbone_type == 'mobilenet_v2':
            # MobileNetV2 backbone
            if pretrained:
                backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
            else:
                backbone = models.mobilenet_v2(pretrained=False)
            
            # Modify the first convolutional layer to accept 6 channels instead of 3
            original_first_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                n_channels, 
                32, 
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias
            )
            
            if pretrained:
                with torch.no_grad():
                    # Copy pretrained weights to the new conv layer
                    # For channels 0-2, copy the original weights
                    backbone.features[0][0].weight[:, 0:3, :, :] = original_first_conv.weight
                    # For channels 3-5, duplicate the original weights (averaging approach)
                    backbone.features[0][0].weight[:, 3:6, :, :] = original_first_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            
            # MobileNetV2 doesn't have the same layer structure as ResNet
            # We need to define our own low_level_features and high_level_features
            # For MobileNetV2, we'll use features from early layers as low-level features
            # and features from later layers as high-level features
            
            # Output stride adjustment for MobileNetV2
            if output_stride == 16:
                # Default MobileNetV2 already has output_stride=16
                pass
            elif output_stride == 8:
                # For MobileNetV2, we need to adjust strides at different layers
                # Change stride from 2 to 1 for key layers to achieve output_stride=8
                
                # Check if the layer has stride attribute and modify it
                layer_indices_to_modify = [6, 13, 14]  # layers with stride=2
                
                for idx in layer_indices_to_modify:
                    if idx < len(backbone.features):
                        layer = backbone.features[idx]
                        if hasattr(layer, 'stride'):
                            # Modify the stride
                            if isinstance(layer.stride, tuple):
                                layer.stride = (1, 1)
                            else:
                                layer.stride = 1
                        
                        # For MobileNetV2 InvertedResidual layers, check child layers
                        if hasattr(layer, 'conv') and hasattr(layer.conv, 'stride'):
                            if isinstance(layer.conv.stride, tuple):
                                layer.conv.stride = (1, 1)
                            else:
                                layer.conv.stride = 1
            
            # Define low-level features (early layers with 24 channels)
            self.low_level_features = nn.Sequential(
                backbone.features[0],  # Conv2d 3->32, stride 2
                backbone.features[1],  # InvertedResidual, stride 1
                backbone.features[2],  # InvertedResidual, stride 2
                backbone.features[3],  # InvertedResidual, stride 1
                backbone.features[4],  # InvertedResidual, stride 1
                backbone.features[5],  # InvertedResidual, stride 1
                backbone.features[6],  # InvertedResidual, stride 2
            )
            
            # Define high-level features (later layers with 1280 channels)
            self.high_level_features = nn.Sequential(
                backbone.features[7],
                backbone.features[8],
                backbone.features[9],
                backbone.features[10],
                backbone.features[11],
                backbone.features[12],
                backbone.features[13],
                backbone.features[14],
                backbone.features[15],
                backbone.features[16],
                backbone.features[17],
                backbone.features[18],
            )
            
            # Update the ASPP input channels for MobileNetV2 (1280 instead of 2048)
            self.aspp = ASPP(1280, 256, [6, 12, 18] if output_stride == 16 else [12, 24, 36])
            
            # Update the reduce_low_level layer for MobileNetV2 (infer actual low-level channels)
            low_level_layer = backbone.features[6]
            low_level_channels = None
            if hasattr(low_level_layer, 'conv'):
                if isinstance(low_level_layer.conv, nn.Sequential):
                    modules = list(low_level_layer.conv)
                else:
                    modules = [low_level_layer.conv]
                for module in reversed(modules):
                    if isinstance(module, nn.Conv2d):
                        low_level_channels = module.out_channels
                        break
            if low_level_channels is None:
                low_level_channels = 24

            self.reduce_low_level = nn.Sequential(
                nn.Conv2d(low_level_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
        else:  # Default to ResNet-50
            if pretrained:
                backbone = models.resnet50(weights="IMAGENET1K_V1")
            else:
                backbone = models.resnet50(pretrained=False)
            
            original_first_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                n_channels, 
                64, 
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias
            )
            
            if pretrained:
                with torch.no_grad():
                    backbone.conv1.weight[:, 0:3, :, :] = original_first_conv.weight
                    backbone.conv1.weight[:, 3:6, :, :] = original_first_conv.weight
            
            if output_stride == 16:
                pass
            elif output_stride == 8:
                backbone.layer4[0].conv1.stride = (1, 1)
                backbone.layer4[0].conv2.stride = (1, 1)
                backbone.layer4[0].downsample[0].stride = (1, 1)
            
            self.low_level_features = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3
            )
            
            self.high_level_features = backbone.layer4
        
        return backbone

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Extract features from backbone
        # low_level_features comes from ResNet-50 layer3 (1024 channels)
        low_level_features = self.low_level_features(x)
        # high_level_features comes from ResNet-50 layer4 (2048 channels)
        high_level_features = self.high_level_features(low_level_features)
        
        # Apply ASPP to high-level features
        aspp_features = self.aspp(high_level_features)
        
        # Upsample ASPP features to match low-level features spatial dimensions
        aspp_features = F.interpolate(
            aspp_features, 
            size=low_level_features.shape[2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # Reduce low-level features channels from 1024 to 256
        low_level_features = self.reduce_low_level(low_level_features)
        
        # Concatenate ASPP features (256 channels) and reduced low-level features (256 channels)
        concat_features = torch.cat([aspp_features, low_level_features], dim=1)
        
        # Apply decoder
        decoder_features = self.decoder(concat_features)
        
        # Final upsampling to input size
        output = F.interpolate(
            decoder_features, 
            size=input_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Apply classifier to get segmentation mask
        output = self.classifier(output)
        
        return output

    def _init_weights(self):
        for m in self.reduce_low_level.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)


def get_deeplabv3_plus_model(n_channels=6, n_classes=1, output_stride=16, pretrained_backbone=True, backbone_type='resnet50'):
    return DeepLabV3Plus(
        n_channels=n_channels,
        n_classes=n_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        backbone_type=backbone_type
    )
