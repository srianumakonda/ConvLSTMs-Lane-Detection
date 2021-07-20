import torch
import torch.nn as nn
import torchvision.models as models
from utils import *

class VGG16UNetLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, input_dim, n_layers, loc, pretrained=False):

        """
        @param:
            pretrained (bool): specify whether the resnet backbone is to be pretrained or not
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
        """

        super(VGG16UNetLSTM,self).__init__()

        self.loc = loc
        self.encoder = models.vgg16_bn(pretrained=pretrained).features
        self.encoder[0] = nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1)
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])
        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.d_bottleneck = double_conv(512,1024)
        self.upconv1 = upsample(1024,512,1024)
        self.upconv2 = upsample(512,256,768)
        self.upconv3 = upsample(256,128,384)
        self.upconv4 = upsample(128,64,192)
        self.upconv5 = upsample(64,32,96)
        self.out = nn.Conv2d(32,out_channels,kernel_size=1)
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=[input_dim]*n_layers,
                                 kernel_size=3,
                                 num_layers=n_layers,
                                 bias=True)

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):

        x  = torch.unbind(x, dim=1)

        if self.loc == "bridge":
            data = []
            for i in x:
                x1 = self.block1(i)
                x2 = self.block2(x1)
                x3 = self.block3(x2)
                x4 = self.block4(x3)
                x5 = self.block5(x4)
                bottleneck = self.bottleneck(x5)
                bottleneck = self.d_bottleneck(bottleneck)
                data.append(bottleneck.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm, _ = self.convlstm(data)
            lstm = lstm[0][-1,:,:,:,:]
            x6 = self.upconv1(lstm,x5)
            x7 = self.upconv2(x6,x4)
            x8 = self.upconv3(x7,x3)
            x9 = self.upconv4(x8,x2)
            x10 = self.upconv5(x9,x1)
            out = self.out(x10)
            out = torch.sigmoid(out)
            return out

        elif self.loc == "end":
            data = []
            for i in x:
                x1 = self.block1(i)
                x2 = self.block2(x1)
                x3 = self.block3(x2)
                x4 = self.block4(x3)
                x5 = self.block5(x4)
                bottleneck = self.bottleneck(x5)
                bottleneck = self.d_bottleneck(bottleneck)
                data.append(bottleneck.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm_data = []
            for i in data:
                x6 = self.upconv1(i,x5)
                x7 = self.upconv2(x6,x4)
                x8 = self.upconv3(x7,x3)
                x9 = self.upconv4(x8,x2)
                x10 = self.upconv5(x9,x1)
                out = self.out(x10)
                out = torch.sigmoid(out)
                lstm_data.append(out.unsqueeze(0))
            lstm_data = torch.cat(lstm_data, dim=0)
            lstm, _ = self.convlstm(lstm_data)
            lstm = lstm[0][-1,:,:,:,:]
            return lstm 