import torch.nn as nn
import torchvision
import torch
from typing import Dict, Tuple


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):
        super(ContextUnet, self).__init__()

        #num of input channels
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        #initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        #initialize the doensampling path
        self.down1 = UnetDown(n=n_feat, N=n_feat)
        self.down1 = UnetDown(n=n_feat, N=n_feat)

        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.GELU())

        #Embed the timestamp and context labels with a one layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed1 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_feat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_feat, 1*n_feat)

        # Initialize the upsampling of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.Conv2d(2*n_feat, 2*n_feat,kernel_size=self.h//4, stride=self.h//4) ,#upsample
            nn.GroupNorm(8, n_feat), #ormalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, kernel_size=3,stride=1,padding=1) #same as input
        )

    def forward(self, x, t, c=None):
        """Implement the forward of the unet
        time embedding to determine the additional noise 
-            context embedding used to enable control in the generation.

        Args:
            x (_type_): input image
            t (_type_): time step
            c (_type_, optional): context label.
        """
        x = self.init_conv(x)
        #Downsampling path]
        down1 = self.down1(x)
        down2 = self.down2(down1)

        #Convert feature maps into vectors and apply convolution
        hiddenvec = self.to_vec(down2)

        # mask out of context if context mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 1, 1)
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2) #add and multiply embeddings]
        up3 = self.up2(cemb2*up2 + temb1, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out






