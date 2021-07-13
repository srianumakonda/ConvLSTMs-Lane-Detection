import torch
import torch.nn as nn
import torchvision.models as models
from utils import double_conv

class VGG16_UNet(nn.Module):

    def __init__(self,pretrained,in_channels,out_channels):

        super(VGG16_UNet,self).__init__()

        self.encoder = models.vgg16_bn(pretrained=pretrained).features
        self.encoder[0] = nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1)

        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.d_bottleneck = double_conv(512,1024)

        self.d_conv1 = double_conv(1024,512)
        self.d_conv2 = double_conv(512+256,256)
        self.d_conv3 = double_conv(256+128,128)
        self.d_conv4 = double_conv(128+64,64)
        self.d_conv5= double_conv(64+32,32)

        self.upconv1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.upconv2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.upconv3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.upconv4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.upconv5 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)

        self.output_conv = nn.Conv2d(32,out_channels,kernel_size=1)

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

        bottleneck = self.bottleneck(block5)
        x = self.d_bottleneck(bottleneck)

        x = self.upconv1(x)
        x = torch.cat([x, block5], dim=1)
        x = self.d_conv1(x)

        x = self.upconv2(x)
        x = torch.cat([x, block4],dim=1)
        x = self.d_conv2(x)

        x = self.upconv3(x)
        x = torch.cat([x, block3],dim=1)
        x = self.d_conv3(x)

        x = self.upconv4(x)
        x = torch.cat([x, block2],dim=1)
        x = self.d_conv4(x)

        x = self.upconv5(x)
        x = torch.cat([x, block1],dim=1)
        x = self.d_conv5(x)

        x = self.output_conv(x)
        x = torch.sigmoid(x)
        return x


