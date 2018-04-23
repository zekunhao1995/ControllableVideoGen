import torch
from torch.autograd import Variable

# Inherit from Function
class CoolTanH(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.clamp(input, min=0., max=1.)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        # input > 1 & grad < 0 --> grad = grad
        # input > 1 & grad > 0 --> grad = 0
        # input < 0 & grad > 0 --> grad = grad
        # input < 0 & grad < 0 --> grad = 0
        grad_gtz = grad_output < 0.
        passcond = ((input > 1.)&(grad_gtz^1)) | ((input < 0.)&grad_gtz)
        grad_input = grad_output*(passcond.type(torch.cuda.FloatTensor))
        return grad_input
        
#cooltanh = CoolTanH.apply
