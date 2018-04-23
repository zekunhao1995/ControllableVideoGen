import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Laplace2D(nn.Module):
    def __init__(self):
        super(Laplace2D, self).__init__()
        t_kx = torch.FloatTensor([[1,0,-1],[2,0,-2],[1,0,-1]]) # 3 3
        sobel_kx = t_kx.expand([2, 1, 3, 3])
        t_ky = torch.FloatTensor([[1,2,1],[0,0,0],[-1,-2,-1]])
        sobel_ky = t_ky.expand([2, 1, 3, 3])
        self.register_buffer('sobel_kx', sobel_kx)
        self.register_buffer('sobel_ky', sobel_ky)
        
    def forward(self, x):
        # x: N 2 W H optical flow
        # weight – filters tensor (out_channels, in_channels/groups, kH, kW)
        num_out = x.size()[2]*x.size()[3]
        dx = F.conv2d(x, Variable(self.sobel_kx), bias=None, stride=1, padding=0, dilation=1, groups=2)
        dy = F.conv2d(x, Variable(self.sobel_ky), bias=None, stride=1, padding=0, dilation=1, groups=2) # N 2 H-2 W-2
        #dx2sumsq = torch.sqrt(torch.sum(torch.mul(dx,dx),1)+0.0001) # N H-2 W-2
        #dy2sumsq = torch.sqrt(torch.sum(torch.mul(dy,dy),1)+0.0001) # N H-2 W-2
        #loss = torch.sum((dx2sumsq+dy2sumsq).view(-1))/num_out
        loss = (torch.mean(torch.sqrt(torch.abs(dx)+1e-5))+torch.mean(torch.sqrt(torch.abs(dy)+1e-5)))*0.5
        return loss

