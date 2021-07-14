from utils.convlstm import ConvLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class UNetConvLSTM(nn.Module):
    
    def __init__(self, in_channels, out_channels, input_dim, n_layers, loc):
        
        """
        @param:
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
            input dim (int): specify the input dimension for the ConvLSTM layer
            n_layers (int): specify the amount of ConvLSTM layers to be placed
            loc (str): specify whether the ConvLSTM is to be placed as bridge between the encoder and decoder or at the end
        """

        super(UNetConvLSTM, self).__init__()

        self.loc = loc
        self.down1 = double_conv(in_channels,64)
        self.down2 = max_down(64,128)
        self.down3 = max_down(128,256)
        self.down4 = max_down(256,512)
        self.down5 = max_down(512,512)
        self.up1 = upsample(1024,256)
        self.up2 = upsample(512,128)
        self.up3 = upsample(256,64)
        self.up4 = upsample(128,64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=[input_dim]*n_layers,
                                 kernel_size=3,
                                 num_layers=n_layers,
                                 bias=True)
        
    def forward(self, x):

        if self.loc == "bridge":
            x  = torch.unbind(x, dim=1)
            data = []
            for i in x:
                x1 = self.down1(i)
                x2 = self.down2(x1)
                x3 = self.down3(x2)
                x4 = self.down4(x3)
                x5 = self.down5(x4)
                data.append(x5.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm, _ = self.convlstm(data)
            lstm = lstm[0][-1,:,:,:,:]
            x = self.up1(lstm,x4)
            x = self.up2(x,x3)
            x = self.up3(x,x2)
            x = self.up4(x,x1)
            out = self.out_conv(x)
            out = torch.sigmoid(out)
            return out  

        elif self.loc == "end":
            x  = torch.unbind(x, dim=1)
            data = []
            for i in x:
                x1 = self.down1(i)
                x2 = self.down2(x1)
                x3 = self.down3(x2)
                x4 = self.down4(x3)
                x5 = self.down5(x4)
                data.append(x5.unsqueeze(0))
            data = torch.cat(data, dim=0)
            lstm_data = []
            for i in data:
                x6 = self.up1(i,x4)
                x7 = self.up2(x6,x3)
                x8 = self.up3(x7,x2)
                x9 = self.up4(x8,x1)
                out = self.out_conv(x9)
                out = torch.sigmoid(out)
                lstm_data.append(out.unsqueeze(0))
            lstm_data = torch.cat(lstm_data, dim=0)
            lstm, _ = self.convlstm(lstm_data)
            lstm = lstm[0][-1,:,:,:,:]
            return lstm      

        else:
            raise KeyError("Invalid input argument for loc parameter")