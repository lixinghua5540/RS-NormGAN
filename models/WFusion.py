"""
Copyright (c) 2023 Jianhao Miao, School of Remote Sensing, Wuhan University.
"""

import torch
import torch.nn as nn
import numpy as np
import functools
import torch.nn.functional as F



class Weightfusion(nn.Module):
    def __init__(self,radius):
        """
        F represents dilated foreground, 
        F_mask is the accurate mask of foreground,
        B represents Background invariant area,
        layer: the edge length of fusion process the number of single erode or dilate 
        """
        super(Weightfusion,self).__init__()
        self.radius = radius
        self.layers=2*self.radius
        self.weight=[]
        for i in range(self.layers):#from small to big
            self.weight.append((i+1)/(self.layers+1))
    
    def dilate(self,bin_img, ksize=3):
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')#pad 有四个参数，分别代表甚么
        out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        return out
        
    def erode(self,bin_img, ksize=3):
        out = 1 - self.dilate(1 - bin_img, ksize)
        return out

    def forward(self,B_img,V_img,V_mask):
        """
        F represents dilated foreground, 
        F_mask is the accurate mask of foreground,
        B represents Background invariant area,
        radius: the edge length of fusion process
        """
        # a fusion method for two generated results
        Fusion_result=torch.zeros_like(B_img)
        indsinv=(V_mask==0)
        indsinv3=torch.cat([indsinv,indsinv,indsinv],axis=1)
        Fusion_result[indsinv3]=B_img[indsinv3]
        inds=(V_mask!=0)#vmask!0是变化v的要用 netG
        inds3=torch.cat([inds,inds,inds],axis=1)
        Fusion_result[inds3]=V_img[inds3]
        Area_resulte=[]
        Area_resultd=[]
        Edge_result=[]
        maske=V_mask
        maskd=V_mask
        Area_resulte.append(V_mask)#former is bigger
        Area_resultd.append(V_mask)#latter is bigger
        for i in range(self.radius):#一步一步的扩张会耗时间感觉，提出一个高效算法一步扩张权重,哪些不要梯度也可以直接取消梯度
            maske=self.erode(maske)#inplace？
            maskd=self.dilate(maskd)
            Area_resulte.append(maske)
            Area_resultd.append(maskd)

        for i in range(self.radius):#from outer to inner
            Edge_result.append(Area_resulte[i]-Area_resulte[i+1])
        #reverse the direction of erode part
        Edge_result.reverse()
        for i in range(self.radius):#from inner to outer
            Edge_result.append(Area_resultd[i+1]-Area_resultd[i])
        for i in range(self.layers):#weight is the weight of bg,from inner to outer
            Fusion_edgeresult=self.weight[i]*Edge_result[i]*B_img+self.weight[self.layers-i-1]*Edge_result[i]*V_img
            inds_edge=(Edge_result[i]!=0)
            inds_edge3=torch.cat([inds_edge,inds_edge,inds_edge],axis=1)
            Fusion_result[inds_edge3]=Fusion_edgeresult[inds_edge3]

        return Fusion_result
