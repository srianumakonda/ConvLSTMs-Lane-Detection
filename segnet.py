import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet_Model(nn.Module):
    
    def __init__(self, in_channels, out_channels, affine, track_running_stats):
        
        """
        Implementation of, "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation" (Badrinarayanan et al., 2017)
        https://arxiv.org/pdf/1511.00561.pdf
        
        @param:
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
            padding (bool): specify if padding is to be (or not) used in the U-Net implementation
            affine (bool): specify if the U-Net model should have learnable affine parameters
            track_running_stats (bool): specify if the U-Net model should be tracking the mean and variance
            
        @note:
            The paper, "Normalization in Training U-Net for 2D Biomedical Semantic Segmentation" (Zhou et al., 2018)  states that the
            use of Instance Normalization can create higher accuracy than other state-of-the-art methods such as Batch Normalization and
            Layer Normalization to help combat the epxloding gradients problem in U-Net network systems.
            
            I choose to use Instance Normalization (IN) before every time ReLU is applied on top of a convolution instead of the typical BN
            in the proposed SegNet model. This will now allow for proper normalization to occur throughout training and have all the numbers
            perfectly centered for the ReLU function. I also choose to set the IN to have learnable affine parameters and allow it to track 
            the running mean and variance instead of choosing to use batch statistics for training and evaluation modes.
        """

        super(SegNet_Model, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.relu = nn.ReLU()
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        self.deconv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        self.deconv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv4_1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        
        self.deconv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.deconv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.deconv3_1 = nn.Conv2d(256, 128, 3, stride=1, padding=1)

        self.deconv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.deconv2_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)

        self.deconv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.deconv1_1 = nn.Conv2d(64, out_channels, 3, stride=1, padding=1)
        
        self.in_64 = nn.InstanceNorm2d(64,affine=affine,track_running_stats=track_running_stats)
        self.in_128 = nn.InstanceNorm2d(128,affine=affine,track_running_stats=track_running_stats)
        self.in_256 = nn.InstanceNorm2d(256,affine=affine,track_running_stats=track_running_stats)
        self.in_512 = nn.InstanceNorm2d(512,affine=affine,track_running_stats=track_running_stats)
        
    def forward(self, x):
        
        conv1_1 = self.relu(self.in_64(self.conv1_1(x)))
        conv1_2 = self.relu(self.in_64(self.conv1_2(conv1_1)))
        pool_1, id1 = F.max_pool2d(conv1_2,kernel_size=2,stride=2,return_indices=True)
        
        conv2_1 = self.relu(self.in_128(self.conv2_1(pool_1)))
        conv2_2 = self.relu(self.in_128(self.conv2_2(conv2_1)))
        pool_2, id2 = F.max_pool2d(conv2_2,kernel_size=2,stride=2,return_indices=True)
        
        conv3_1 = self.relu(self.in_256(self.conv3_1(pool_2)))
        conv3_2 = self.relu(self.in_256(self.conv3_2(conv3_1)))
        conv3_3 = self.relu(self.in_256(self.conv3_3(conv3_2)))
        pool_3, id3 = F.max_pool2d(conv3_3,kernel_size=2,stride=2,return_indices=True)
        
        conv4_1 = self.relu(self.in_512(self.conv4_1(pool_3)))
        conv4_2 = self.relu(self.in_512(self.conv4_2(conv4_1)))
        conv4_3 = self.relu(self.in_512(self.conv4_3(conv4_2)))
        pool_4, id4 = F.max_pool2d(conv4_3,kernel_size=2,stride=2,return_indices=True)
        
        conv5_1 = self.relu(self.in_512(self.conv5_1(pool_4)))
        conv5_2 = self.relu(self.in_512(self.conv5_2(conv5_1)))
        conv5_3 = self.relu(self.in_512(self.conv5_3(conv5_2)))
        pool_5, id5 = F.max_pool2d(conv5_3,kernel_size=2,stride=2,return_indices=True)
        
        unpool_5 = F.max_unpool2d(pool_5,id5,kernel_size=2,stride=2)
        deconv5_3 = self.relu(self.in_512(self.deconv5_3(unpool_5)))
        deconv5_2 = self.relu(self.in_512(self.deconv5_2(deconv5_3)))
        deconv5_1 = self.relu(self.in_512(self.deconv5_1(deconv5_2)))
        
        unpool_4 = F.max_unpool2d(deconv5_1,id4,kernel_size=2,stride=2)
        deconv4_3 = self.relu(self.in_512(self.deconv4_3(unpool_4)))
        deconv4_2 = self.relu(self.in_512(self.deconv4_2(deconv4_3)))
        deconv4_1 = self.relu(self.in_256(self.deconv4_1(deconv4_2)))
        
        unpool_3 = F.max_unpool2d(deconv4_1,id3,kernel_size=2,stride=2)
        deconv3_3 = self.relu(self.in_256(self.deconv3_3(unpool_3)))
        deconv3_2 = self.relu(self.in_256(self.deconv3_2(deconv3_3)))
        deconv3_1 = self.relu(self.in_128(self.deconv3_1(deconv3_2)))
        
        unpool_2 = F.max_unpool2d(deconv3_1,id2,kernel_size=2,stride=2)
        deconv2_2 = self.relu(self.in_128(self.deconv2_2(unpool_2)))
        deconv2_1 = self.relu(self.in_64(self.deconv2_1(deconv2_2)))
        
        unpool_1 = F.max_unpool2d(deconv2_1,id1,kernel_size=2,stride=2)
        deconv1_2 = self.relu(self.in_64(self.deconv1_2(unpool_1)))
        output = self.deconv1_1(deconv1_2)
        
        output = torch.sigmoid(output)
        return output