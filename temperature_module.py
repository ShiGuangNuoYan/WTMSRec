import math
from re import X
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
import torch

class Global_t(nn.Module):
    def __init__(self):
        super(Global_t, self).__init__()

        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, lambda_):
        return self.grl(self.global_T, lambda_)


from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)

# class ChangeT(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#
#     def __init__(self):
#         super(ChangeT, self).__init__()
#
#     def forward(self, y_s, y_t, temp):
#         T = temp.cuda()
#
#         KD_loss = 0
#         KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_s / T, dim=1),
#                                                        F.softmax(y_t / T, dim=1)) * T * T
#         return KD_loss
class ChangeT(nn.Module):
    def __init__(self):
        super(ChangeT, self).__init__()

    def forward(self, logits, batch_labels, temp):
        T = temp.to(logits.device)

        # Scale logits by temperature
        scaled_logits = logits / T

        # Compute the cross-entropy loss
        CE_loss = F.cross_entropy(scaled_logits, batch_labels) 

        return CE_loss

