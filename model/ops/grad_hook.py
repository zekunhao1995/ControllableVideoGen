import torch
from torch.autograd import Variable

# Inherit from Function
class CoolTanH(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)

        return input, ctx.saved_variables[0]

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
    
        input = ctx.saved_variables[0]
        
        return grad_output
        
#cooltanh = CoolTanH.apply
