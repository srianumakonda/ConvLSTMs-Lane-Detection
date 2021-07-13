import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_Model(nn.Module):
    
    def __init__(self, in_channels, out_channels, affine, track_running_stats):
        
        """
        Implementation of, "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        
        @param:
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
            affine (bool): specify if the U-Net model should have learnable affine parameters
            track_running_stats (bool): specify if the U-Net model should be tracking the mean and variance
            
        
        @note: 
            I'm using padding = 1 while the paper does not. This is easily noticed as the paper has a different output 
            size (388 x 388) than the input amount of pixels (572 x 572). Because in the case of lane detection, I'd need 
            an output that is the same size as the input, especially with respect to testing + validation.
            
        @note:
            I choose to use torch.nn.Upsample() instead of torch.nn.ConvTranspose2D() mainly because this does fit in
            with what the paper describes in Section 2:
            
                Every step in the expansive path consists of an upsampling of the
                feature map followed by a 2x2 convolution (“up-convolution”) that halves the
                number of feature channels, a concatenation with the correspondingly cropped
                feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.
                
            Because of this, I choose to use torch.nn.Upsample(). The main difference between that and 
            torch.nn.ConvTranspose2D() is that the former does not use any sort of learning parameters
            vs. the latter which does require weight updates + calculation of the gradients. 
            
            I make the assumption that in the self-driving environment (which is extremely dynamic), I would rather
            go for a faster interface than the latter. According to this study from https://github.com/jvanvugt/pytorch-unet/issues/1,
            the user states that there is minimal difference between the use of both deconvolution techniques.
            
        @note:
            The paper, "Normalization in Training U-Net for 2D Biomedical Semantic Segmentation" (Zhou et al., 2018)  states that the
            use of Instance Normalization can create higher accuracy than other state-of-the-art methods such as Batch Normalization and
            Layer Normalization to help combat the epxloding gradients problem in U-Net network systems.
            
            I choose to use Instance Normalization (IN) after every time ReLU is applied on top of a convolution. This will now allow for
            proper normalization to occur throughout training. I also choose to set the IN to have learnable affine parameters and allow
            it to track the running mean and variance instead of choosing to use batch statistics for training and evaluation modes.
        """

        super(UNet_Model, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        
        self.deconv1 = nn.Conv2d(1024+512, 512, 3, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv3 = nn.Conv2d(512+256, 256, 3, stride=1, padding=1)
        self.deconv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.deconv5 = nn.Conv2d(256+128, 128, 3, stride=1, padding=1)
        self.deconv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.deconv7 = nn.Conv2d(128+64, 64, 3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.deconv9 = nn.Conv2d(64, self.out_channels, 1, stride=1)
        
        self.in_64 = nn.InstanceNorm2d(64,affine=affine,track_running_stats=track_running_stats)
        self.in_128 = nn.InstanceNorm2d(128,affine=affine,track_running_stats=track_running_stats)
        self.in_256 = nn.InstanceNorm2d(256,affine=affine,track_running_stats=track_running_stats)
        self.in_512 = nn.InstanceNorm2d(512,affine=affine,track_running_stats=track_running_stats)
        self.in_1024 = nn.InstanceNorm2d(1024,affine=affine,track_running_stats=track_running_stats)
        
    def forward(self, x):
        
        conv_block_1 = F.relu(self.in_64(self.conv1(x)))
        conv_block_1 = F.relu(self.in_64(self.conv2(conv_block_1)))
        conv_block_1_max = self.maxpool(conv_block_1)
        
        conv_block_2 = F.relu(self.in_128(self.conv3(conv_block_1_max)))
        conv_block_2 = F.relu(self.in_128(self.conv4(conv_block_2)))
        conv_block_2_max = self.maxpool(conv_block_2)
        
        conv_block_3 = F.relu(self.in_256(self.conv5(conv_block_2_max)))
        conv_block_3 = F.relu(self.in_256(self.conv6(conv_block_3)))
        conv_block_3_max = self.maxpool(conv_block_3)
        
        conv_block_4 = F.relu(self.in_512(self.conv7(conv_block_3_max)))
        conv_block_4 = F.relu(self.in_512(self.conv8(conv_block_4)))
        conv_block_4_max = self.maxpool(conv_block_4)
        
        conv_block_5 = F.relu(self.in_1024(self.conv9(conv_block_4_max)))
        conv_block_5 = F.relu(self.in_1024(self.conv10(conv_block_5)))
        
        upconv_block_1 = self.in_1024(self.upsample(conv_block_5))
        upconv_block_1 = torch.cat([upconv_block_1, conv_block_4], dim=1)
        upconv_block_1 = F.relu(self.in_512(self.deconv1(upconv_block_1)))
        upconv_block_1 = F.relu(self.in_512(self.deconv2(upconv_block_1)))
        
        upconv_block_2 = self.in_512(self.upsample(upconv_block_1))
        upconv_block_2 = torch.cat([upconv_block_2, conv_block_3], dim=1)
        upconv_block_2 = F.relu(self.in_256(self.deconv3(upconv_block_2)))
        upconv_block_2 = F.relu(self.in_256(self.deconv4(upconv_block_2)))
        
        upconv_block_3 = self.in_256(self.upsample(upconv_block_2))
        upconv_block_3 = torch.cat([upconv_block_3, conv_block_2], dim=1)
        upconv_block_3 = F.relu(self.in_128(self.deconv5(upconv_block_3)))
        upconv_block_3 = F.relu(self.in_128(self.deconv6(upconv_block_3)))

        upconv_block_4 = self.in_128(self.upsample(upconv_block_3))
        upconv_block_4 = torch.cat([upconv_block_4, conv_block_1], dim=1)
        upconv_block_4 = F.relu(self.in_64(self.deconv7(upconv_block_4)))
        upconv_block_4 = F.relu(self.in_64(self.deconv8(upconv_block_4)))
        
        out = self.deconv9(upconv_block_4)
        out = torch.sigmoid(out)
        return out        