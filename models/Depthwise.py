"""
Copyright (c) 2024 Jianhao Miao, School of Remote Sensing, Wuhan University.
"""

import torch
import torch.nn as nn
import numpy as np
import functools

class DWSC_unit(nn.Module):#the basic module of DWSC
    def __init__(self, in_ch, out_ch, kernel_size, padding, groups):
        super(DWSC_unit, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,#The stride of DWSC is set to 1
            padding=padding,
            groups=groups,
            bias=False
        )

    def forward(self, input):
        out = self.conv(input)
        return out

class DWSC(nn.Module):#Depthwise Separable Convolution
    def __init__(self, in_ch, out_ch, kernel_size, padding,norm):
        """
        :ksize define the  the kernel size of channel wise convolution，
        groups:the channel number of channel wise convolution
        """
        #model=[]
        super(DWSC, self).__init__()
        self.depthwiseconv = DWSC_unit(in_ch,in_ch,kernel_size,padding=padding,groups=in_ch)
        self.pointwiseconv = DWSC_unit(in_ch,out_ch,kernel_size=1,padding=0,groups=1)
        self.relu1=nn.PReLU()
        self.relu2=nn.PReLU()

        if norm=="BatchNorm2d":
            self.norm1=nn.BatchNorm2d(num_features=in_ch)
            self.norm2=nn.BatchNorm2d(num_features=out_ch)
        else:
            self.norm1=nn.InstanceNorm2d(num_features=in_ch)
            self.norm2=nn.InstanceNorm2d(num_features=in_ch)
        #self.process=nn.sequential()
        
    def forward(self, input):
        dwr=self.depthwiseconv(input)
        dwr=self.relu1(self.norm1(dwr))
        pwr=self.pointwiseconv(dwr)
        #out = self.relu2(self.norm2(pwr))
        return pwr


# class UpsampleConvLayer(torch.nn.Module):
#     """UpsampleConvLayer
#     Upsamples the input and then does a convolution. This method gives better results
#     compared to ConvTranspose2d.
#     ref: http://distill.pub/2016/deconv-checkerboard/
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super(UpsampleConvLayer, self).__init__()
#         self.upsample = upsample
#         if upsample:
#             self.upsample_layer = torch.nn.Upsample(scale_factor=upsample,mode="bilinear",align_corners=True)
#         self.reflection_padding = int(np.floor(kernel_size / 2))
#         if self.reflection_padding != 0:
#             self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
#         #self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#         self.conv2d=DWSC(in_channels,out_channels,kernel_size,padding=0,norm="InstanceNorm2d")#??
#     def forward(self, x):
#         if self.upsample:
#             x = self.upsample_layer(x)
#         if self.reflection_padding != 0:
#             x = self.reflection_pad(x)
#         out = self.conv2d(x)
#         return out