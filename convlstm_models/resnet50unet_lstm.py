import torch
import torch.nn as nn
import torchvision.models as models
from utils import *

class ResNet50UNetLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, input_dim, n_layers, loc, pretrained=False):

        """
        @param:
            pretrained (bool): specify whether the resnet backbone is to be pretrained or not
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
        """

        super(ResNet50UNetLSTM,self).__init__()

        self.loc = loc
        self.encoder = models.resnet50(pretrained=pretrained)
        self.encoder.conv1 = nn.Conv2d(in_channels,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.encoder_layers = list(self.encoder.children())
        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]
        self.upconv1 = upsample(2048,512,1536)
        self.upconv2 = upsample(512,256,768)
        self.upconv3 = upsample(256,128,384)
        self.upconv4 = upsample(128,64,128)
        self.upconv5 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)

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
                data.append(x5.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm, _ = self.convlstm(data)
            lstm = lstm[0][-1,:,:,:,:]
            x6 = self.upconv1(lstm,x4)
            x7 = self.upconv2(x6,x3)
            x8 = self.upconv3(x7,x2)
            x9 = self.upconv4(x8,x1)
            out = self.upconv5(x9)
            out = self.out(out)
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
                data.append(x5.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm_data = []
            for i in data:
                x6 = self.upconv1(i,x4)
                x7 = self.upconv2(x6,x3)
                x8 = self.upconv3(x7,x2)
                x9 = self.upconv4(x8,x1)
                out = self.upconv5(x9)
                out = self.out(out)
                out = torch.sigmoid(out)
                lstm_data.append(out.unsqueeze(0))
            lstm_data = torch.cat(lstm_data, dim=0)
            lstm, _ = self.convlstm(lstm_data)
            lstm = lstm[0][-1,:,:,:,:]
            return lstm    

        else:
            raise KeyError("Invalid input argument for loc parameter")


