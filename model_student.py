import torch
import torch.nn as nn
import torch.nn.functional as F

class UIB_Block_3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4.0):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_c * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv3d(in_c, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv3d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm3d(out_c)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_c + skip_c, skip_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(skip_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MobileNetV4_UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU6(inplace=True)
        )

        self.layer1 = nn.Sequential(
            UIB_Block_3D(32, 32, stride=1, expand_ratio=1),
            UIB_Block_3D(32, 48, stride=2, expand_ratio=4)
        )

        self.layer2 = nn.Sequential(
            UIB_Block_3D(48, 48, stride=1, expand_ratio=4),
            UIB_Block_3D(48, 96, stride=2, expand_ratio=4)
        )

        self.layer3 = nn.Sequential(
            UIB_Block_3D(96, 96, stride=1, expand_ratio=4),
            UIB_Block_3D(96, 128, stride=2, expand_ratio=4),
            UIB_Block_3D(128, 128, stride=1, expand_ratio=4)
        )

        self.up1 = UpBlock(128, 96)  
        self.up2 = UpBlock(96, 48)   
        self.up3 = UpBlock(48, 32)   

        self.final_up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.final_conv = nn.Conv3d(32, num_classes, 1)

    def forward(self, x):
        x0 = self.stem(x)   
        x1 = self.layer1(x0)  
        x2 = self.layer2(x1)  
        x3 = self.layer3(x2)  

        d1 = self.up1(x3, x2)  
        d2 = self.up2(d1, x1)  
        d3 = self.up3(d2, x0)  

        out = self.final_up(d3) 
        logits = self.final_conv(out)

        return logits, [x1, x2, x3]