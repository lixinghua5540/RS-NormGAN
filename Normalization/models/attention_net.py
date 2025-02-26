from cmath import inf, nan
import torch
import itertools
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from models.Depthwise import DWSC

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample,mode="bilinear",align_corners=True)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        #self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv2d=DWSC(in_channels,out_channels,kernel_size,padding=0,norm="InstanceNorm2d")#??
        #self.conv2d=AtrousSeparableConvolution(in_channels,out_channels,kernel_size,stride=1,padding=0)
    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

class AttentionUNet(nn.Module):
    #A downsampling and upsampling structure (Global attention net) to balance the percentage of origin image and changed image
    def __init__(self, in_nc,outer_nc=1,ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=True,use_bias=True):
        super(AttentionUNet, self).__init__()
        self.preconv = nn.Sequential(nn.Conv2d(in_nc, ngf, kernel_size=3,stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf), 
                      nn.LeakyReLU(0.1,True)
        )
        self.conv1 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3,stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf*2), 
                      nn.LeakyReLU(0.1,True)#
        )
        self.conv2 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3,stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf*4), 
                      nn.LeakyReLU(0.1,True)
        )
        self.upsample1=nn.Sequential(UpsampleConvLayer(ngf*4, ngf*4,kernel_size=5,stride=1,upsample=2),
                      norm_layer(ngf*4), 
                      nn.LeakyReLU(0.1,True)
        )
        self.upsample2=nn.Sequential(UpsampleConvLayer(ngf*4, ngf*4,kernel_size=5,stride=1,upsample=2),
                      norm_layer(ngf*4), 
                      nn.LeakyReLU(0.1,True)
        )
        self.upconv1=nn.Sequential(nn.Conv2d(ngf*4,ngf*2,kernel_size=3,stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf*2), 
                      nn.LeakyReLU(0.1,True)
        )
        self.upconv2=nn.Sequential(nn.Conv2d(ngf*4,ngf,kernel_size=3,stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf), 
                      nn.LeakyReLU(0.1,True)
        )
        self.postconv=nn.Sequential(nn.Conv2d(ngf*2, outer_nc, kernel_size=3,stride=1, padding=1, bias=use_bias),
                      nn.Sigmoid()
                      )

    def forward(self, x):
        x1=self.preconv(x)
        #print("x1",x1.shape)
        x2=self.conv1(x1)
        x3=self.conv2(x2)
        x3=self.upconv1(self.upsample1(x3))
        x4=torch.cat([x2,x3],dim=1)
        x4=self.upconv2(self.upsample2(x4))
        x5=torch.cat([x1,x4],dim=1)
        out=self.postconv(x5)
        return out