from torch import nn


class Unflatten(nn.Module):
    def __init__(self, dim, shape):
        super(Unflatten, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
