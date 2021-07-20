import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SegNetLSTM(nn.Module):
    
    def __init__(self, in_channels, out_channels, input_dim, n_layers, loc):
        
        """
        @param:
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
        """

        super(SegNetLSTM, self).__init__()
        
        self.loc = loc
        self.down1 = double_conv(in_channels,64)
        self.down2 = double_conv(64,128)
        self.down3 = nn.Sequential(
            double_conv(128,256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1))
        self.down4 = nn.Sequential(
            double_conv(256,512),
            nn.Conv2d(512, 512, 3, stride=1, padding=1))
        self.down5 = nn.Sequential(
            double_conv(512,512),
            nn.Conv2d(512, 512, 3, stride=1, padding=1))
        self.upconv1 = nn.Sequential(
            double_conv(512,512),
            nn.Conv2d(512, 512, 3, stride=1, padding=1))
        self.upconv2 = nn.Sequential(
            double_conv(512,512),
            nn.Conv2d(512, 256, 3, stride=1, padding=1))
        self.upconv3 = nn.Sequential(
            double_conv(256,256),
            nn.Conv2d(256, 128, 3, stride=1, padding=1))
        self.upconv4 = double_conv(128,64)
        self.upconv5 = double_conv(64,out_channels)
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=[input_dim]*n_layers,
                                 kernel_size=3,
                                 num_layers=n_layers,
                                 bias=True)

    def forward(self, x):
        
        x  = torch.unbind(x, dim=1)

        if self.loc == "bridge":
            data = []
            for i in x:
                x1 = self.down1(i)
                pool_1, id1 = F.max_pool2d(x1,kernel_size=2,stride=2,return_indices=True)
                x2 = self.down2(pool_1)
                pool_2, id2 = F.max_pool2d(x2,kernel_size=2,stride=2,return_indices=True)
                x3 = self.down3(pool_2)
                pool_3, id3 = F.max_pool2d(x3,kernel_size=2,stride=2,return_indices=True)
                x4 = self.down4(pool_3)
                pool_4, id4 = F.max_pool2d(x4,kernel_size=2,stride=2,return_indices=True)
                x5 = self.down5(pool_4)
                pool_5, id5 = F.max_pool2d(x5,kernel_size=2,stride=2,return_indices=True)
                data.append(pool_5.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm, _ = self.convlstm(data)
            lstm = lstm[0][-1,:,:,:,:]
            unpool_1 = F.max_unpool2d(lstm,id5,kernel_size=2,stride=2)
            x6 = self.upconv1(unpool_1)
            unpool_2 = F.max_unpool2d(x6,id4,kernel_size=2,stride=2)
            x7 = self.upconv2(unpool_2)
            unpool_3 = F.max_unpool2d(x7,id3,kernel_size=2,stride=2)
            x8 = self.upconv3(unpool_3)
            unpool_4 = F.max_unpool2d(x8,id2,kernel_size=2,stride=2)
            x9 = self.upconv4(unpool_4)
            unpool_5 = F.max_unpool2d(x9,id1,kernel_size=2,stride=2)
            out = self.upconv5(unpool_5)
            out = torch.sigmoid(out)
            return out

        elif self.loc == "end":
            data = []
            for i in x:
                x1 = self.down1(i)
                pool_1, id1 = F.max_pool2d(x1,kernel_size=2,stride=2,return_indices=True)
                x2 = self.down2(pool_1)
                pool_2, id2 = F.max_pool2d(x2,kernel_size=2,stride=2,return_indices=True)
                x3 = self.down3(pool_2)
                pool_3, id3 = F.max_pool2d(x3,kernel_size=2,stride=2,return_indices=True)
                x4 = self.down4(pool_3)
                pool_4, id4 = F.max_pool2d(x4,kernel_size=2,stride=2,return_indices=True)
                x5 = self.down5(pool_4)
                pool_5, id5 = F.max_pool2d(x5,kernel_size=2,stride=2,return_indices=True)
                data.append(pool_5.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm_data = []
            for i in data:
                unpool_1 = F.max_unpool2d(i,id5,kernel_size=2,stride=2)
                x6 = self.upconv1(unpool_1)
                unpool_2 = F.max_unpool2d(x6,id4,kernel_size=2,stride=2)
                x7 = self.upconv2(unpool_2)
                unpool_3 = F.max_unpool2d(x7,id3,kernel_size=2,stride=2)
                x8 = self.upconv3(unpool_3)
                unpool_4 = F.max_unpool2d(x8,id2,kernel_size=2,stride=2)
                x9 = self.upconv4(unpool_4)
                unpool_5 = F.max_unpool2d(x9,id1,kernel_size=2,stride=2)
                out = self.upconv5(unpool_5)
                out = torch.sigmoid(out)
                lstm_data.append(out.unsqueeze(0))
            lstm_data = torch.cat(lstm_data, dim=0)
            lstm, _ = self.convlstm(lstm_data)
            lstm = lstm[0][-1,:,:,:,:]
            return lstm  
