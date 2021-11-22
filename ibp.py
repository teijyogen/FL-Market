import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
    def interval(self, upper, lower):
        mid = (upper + lower) / 2.0
        diff = (upper - lower) / 2.0

        center = torch.addmm(self.bias, mid, self.weight.t())
        deviation = torch.mm(diff, self.weight.abs().t())
        upper = center + deviation
        lower = center - deviation
        return upper, lower


class ReLUClipped(nn.Module):
    def __init__(self, lower=0, upper=1):
        super(ReLUClipped, self).__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        x = torch.clamp(x, self.lower, self.upper)
        return x

    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)

class ReLU(nn.ReLU):
    def __init__(self):
        super(ReLU, self).__init__()
    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)

class Sigmoid(nn.Sigmoid):
    def __init__(self, k=1):
        super(Sigmoid, self).__init__()
        self.k = k
    def forward(self, x):
        return 1/(1+torch.exp(-self.k*x))
    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)

class SigmoidLinear(nn.Module):
    def __init__(self, mult=1):
        super(SigmoidLinear, self).__init__()
        self.mult = mult

    def forward(self, x):
        output = torch.where((x>-5) & (x<5), x/10+0.5, torch.tensor(0.).to(x.device))
        output += torch.where((x>5), torch.tensor(1.).to(x.device), torch.tensor(0.).to(x.device))
        return output*self.mult
    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)


class Tanh(nn.Tanh):
    def __init__(self):
        super(Tanh, self).__init__()
    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)

class Softmax(nn.Softmax):
    def __init__(self, dim=None):
        super(Softmax, self).__init__(dim=dim)
    def interval(self, upper, lower):
        mask = torch.eye(upper.shape[1], device=upper.device, dtype=torch.bool)[None, :, :, None]
        upper, lower = F.softmax(torch.where(mask, upper[:, None, :, :], lower[:, None, :, :]), dim=2).max(dim=1)[0], \
                       F.softmax(torch.where(mask, lower[:, None, :, :], upper[:, None, :, :]), dim=2).min(dim=1)[0]
        return upper, lower

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)
    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)

class View_Cut(nn.Module):
    def __init__(self):
        super(View_Cut, self).__init__()
    def forward(self, x):
        return x[:, :, :-1]
    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)

class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
    def interval(self, upper, lower):
        for module in self:
            upper, lower = module.interval(upper, lower)
        return upper, lower
    def reg(self, upper, lower):
        reg = 0
        for module in self:
            if isinstance(module, nn.ReLU):
                reg += -torch.tanh(1+upper*lower)
            if isinstance(module, SigmoidLinear):
                break
            upper, lower = module.interval(upper, lower)
        return reg

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    def interval(self, upper, lower):
        return self.forward(upper), self.forward(lower)