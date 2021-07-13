import torch
import torch.nn as nn
import torchvision.models as models
from utils import double_conv

class ResNet50_UNet(nn.Module):

    def __init__(self,pretrained,in_channels,out_channels):

        super(ResNet50_UNet,self).__init__()

        self.encoder = models.resnet50(pretrained=pretrained)
        self.encoder.conv1 = nn.Conv2d(in_channels,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.relu = nn.ReLU()

        self.d_conv1 = double_conv(1024+512,512)
        self.d_conv2 = double_conv(512+256,256)
        self.d_conv3 = double_conv(256+128,128)
        self.d_conv4 = double_conv(64+64,64)
        self.output_conv = nn.Conv2d(32,out_channels,kernel_size=1)

        self.upconv1 = nn.ConvTranspose2d(2048,512,kernel_size=2,stride=2)
        self.upconv2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.upconv3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.upconv4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.upconv5 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)

    # credit for weight_initialization: https://github.com/zhoudaxia233/PyTorch-Unet/blob/master/resnet_unet.py
    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.upconv1(block5)
        x = torch.cat([x, block4],dim=1)
        x = self.d_conv1(x)

        x = self.upconv2(x)
        x = torch.cat([x, block3],dim=1)
        x = self.d_conv2(x)

        x = self.upconv3(x)
        x = torch.cat([x, block2],dim=1)
        x = self.d_conv3(x)

        x = self.upconv4(x)
        x = torch.cat([x, block1],dim=1)
        x = self.d_conv4(x)

        x = self.upconv5(x)
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        return x


