import torch
import torch.nn as nn
from torch.autograd import Variable


class HardshinkLoss(nn.Module):
    def __init__(self, lowbound, upbound):
        super(HardshinkLoss, self).__init__()
        self.lowbound = lowbound
        self.upbound = upbound
        
    def forward(self, input):
        passcond = (input>self.upbound)|(input<self.lowbound)
        pass_input = input*passcond.type_as(input)
        loss = torch.mean(pass_input**2)
        return loss

