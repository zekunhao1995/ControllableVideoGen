import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class FlowWarp(nn.Module):
    def __init__(self):
        super(FlowWarp, self).__init__()
        self.h = -1;
        self.w = -1;
        
    def forward(self, x, f):
        # First, generate absolute coordinate from relative coordinates
        # f: N (rx,ry) oH oW
        # target: N oH oW (ax(width),ay(height))
        
        # Generate offset map
        width = x.size()[3]
        height = x.size()[2]
        if width != self.w or height != self.h:
            width_map = torch.arange(0, width, step=1).expand([height, width])
            height_map = torch.arange(0, height, step=1).unsqueeze(1).expand([height, width])
            self.offset_map = Variable(torch.stack([width_map,height_map],2).cuda())
            self.w = width
            self.h = height
            self.scaler = Variable(1./torch.cuda.FloatTensor([(self.w-1)/2, (self.h-1)/2]))
            
        f = f.permute(0,2,3,1) # N H W C
        f = f + self.offset_map # add with dimension expansion
        f = f * self.scaler - 1 # scale to [-1,1]

        return F.grid_sample(x, f, mode='bilinear') # eltwise multiply with broadcast
        
