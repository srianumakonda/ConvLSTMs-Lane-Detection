import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class ResUNet_Model(nn.Module):
    
    def __init__(self, in_channels, out_channels, affine, track_running_stats):
        
        """
        Implementation of, "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data" (Diakogiannis et al., 2019)
        https://arxiv.org/pdf/1904.00592.pdf
        
        @param:
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
            affine (bool): specify if the U-Net model should have learnable affine parameters
            track_running_stats (bool): specify if the U-Net model should be tracking the mean and variance
        
        @note:
            The paper, "Normalization in Training U-Net for 2D Biomedical Semantic Segmentation" (Zhou et al., 2018)  states that the
            use of Instance Normalization can create higher accuracy than other state-of-the-art methods such as Batch Normalization and
            Layer Normalization to help combat the epxloding gradients problem in U-Net network systems.

            I choose to use Instance Normalization (IN) after every time ReLU is applied on top of a convolution. This will now allow for
            proper normalization to occur throughout training. I also choose to set the IN to have learnable affine parameters and allow
            it to track the running mean and variance instead of choosing to use batch statistics for training and evaluation modes.
            
        @note:
            According to the Deep Learning Book (Goodfellow et al., 2016), batch normalization should be used after an activation is applied.
            This is the modification that I make to the original Resunet implementation to ensure that I can get the highest possible outputs
            for the segmentation task.
        """

        super(ResUNet_Model, self).__init__()
        
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        
        self.res_block_1 = Residual_Block(64, 128, stride=2,affine=affine,track_running_stats=track_running_stats)
        self.res_block_2 = Residual_Block(128, 256, stride=2,affine=affine,track_running_stats=track_running_stats)
        self.res_block_3 = Residual_Block(256, 512, stride=2,affine=affine,track_running_stats=track_running_stats)

        self.dec_block_1 = Decoder_Block(512, 256,affine=affine,track_running_stats=track_running_stats)
        self.dec_block_2 = Decoder_Block(256, 128,affine=affine,track_running_stats=track_running_stats)
        self.dec_block_3 = Decoder_Block(128, 64,affine=affine,track_running_stats=track_running_stats)

        self.in_1 = nn.InstanceNorm2d(64,affine=affine,track_running_stats=track_running_stats)
        

    def forward(self, x):
        
        enc_1 = self.relu(self.in_1(self.conv1(x)))
        enc_1 = self.conv2(enc_1)
        s = self.conv3(x)
        skip_1 = x + s

        skip_2 = self.res_block_1(skip_1)
        skip_3 = self.res_block_2(skip_2)
        
        bridge = self.res_block_3(skip_3)

        dec_block_1 = self.dec_block_1(bridge, skip_3)
        dec_block_2 = self.dec_block_2(dec_block_1, skip_2)
        dec_block_3 = self.dec_block_3(dec_block_2, skip_1)

        output = self.conv4(dec_block_3)
        output = torch.sigmoid(output)

        return output