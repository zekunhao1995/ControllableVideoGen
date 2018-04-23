import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


class VGG_enc(nn.Module):
    def __init__(self, input_channels=6):
        super(VGG_enc, self).__init__()
        in_channels = input_channels
        self.c11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.c22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.c32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.c33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.c42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.c43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.c52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.c53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        o11 = F.relu(self.c11(x), inplace=True)
        o12 = F.relu(self.c12(o11), inplace=True)
        o1p = self.p1(o12)
        o21 = F.relu(self.c21(o1p), inplace=True)
        o22 = F.relu(self.c22(o21), inplace=True)
        o2p = self.p2(o22)
        o31 = F.relu(self.c31(o2p), inplace=True)
        o32 = F.relu(self.c32(o31), inplace=True)
        o33 = F.relu(self.c33(o32), inplace=True)
        o3p = self.p3(o33)
        o41 = F.relu(self.c41(o3p), inplace=True)
        o42 = F.relu(self.c42(o41), inplace=True)
        o43 = F.relu(self.c43(o42), inplace=True)
        o4p = self.p4(o43)
        o51 = F.relu(self.c51(o4p), inplace=True)
        o52 = F.relu(self.c52(o51), inplace=True)
        o53 = F.relu(self.c53(o52), inplace=True)
        return o53, o43, o33
        
class VGG_dec(nn.Module):
    def __init__(self):
        super(VGG_dec, self).__init__()
        out_channels = 6
        self.c53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)
        self.c52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.c51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.u5 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.c43 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)
        self.c42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.c41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(256)
        self.u4 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.c33 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)
        self.c32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.c31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(128)
        self.u3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.c22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.c21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(64)
        self.u2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        #self.c11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.bn11 = nn.BatchNorm2d(64)
        
        
    def forward(self, i53, i43, i33):
        o53 = F.relu(self.c53(i53), inplace=True)
        o52 = F.relu(self.c52(o53), inplace=True)
        o51 = F.relu(self.c51(o52), inplace=True)
        o5u = self.u5(o51)
        o5c = torch.cat((o5u, i43), 1)
    
        o43 = F.relu(self.c43(o5c), inplace=True)
        o42 = F.relu(self.c42(o43), inplace=True)
        o41 = F.relu(self.c41(o42), inplace=True)
        o4u = self.u4(o41)
        o4c = torch.cat((o4u, i33), 1)
        
        o33 = F.relu(self.c33(o4c), inplace=True)
        o32 = F.relu(self.c32(o33), inplace=True)
        o31 = F.relu(self.c31(o32), inplace=True)
        o3u = self.u3(o31)
        
        o22 = F.relu(self.c22(o3u), inplace=True)
        o21 = F.relu(self.c21(o22), inplace=True)
        o2u = self.u2(o21)
        
        o12 = F.relu(self.c12(o2u), inplace=True)
        #o11 = F.relu(self.bn11(self.c11(o12)), inplace=True)
        
        return o12
        
class VGG_net(nn.Module):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    def __init__(self, input_channels):
        super(VGG_net, self).__init__()
        self.enc_net = VGG_enc(input_channels)
        self.dec_net = VGG_dec()
        self.conv_warp = nn.Conv2d(self.cfg[0], 2, kernel_size=3, padding=1)
        self.conv_mask = nn.Conv2d(self.cfg[0], 1, kernel_size=3, padding=1)
        self.conv_comp = nn.Conv2d(self.cfg[0], 3, kernel_size=3, padding=1)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # input: Nx3x3x256x320
    def forward(self, x):
        dec_feat = self.dec_net(*self.enc_net(x))
        flow = self.conv_warp(dec_feat)
        mask = self.conv_mask(dec_feat)
        comp = self.conv_comp(dec_feat)
        return flow, mask, comp

        

def VGG_Warper(input_channels = 6):
    return VGG_net(input_channels)
