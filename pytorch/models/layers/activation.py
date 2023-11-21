#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

from torch import nn
from typing import Literal

type_activation = Literal["relu", "lrelu", "softmax", "tanh"]


def get_activation_fn(name: type_activation, inplace=True):
    if name.casefold() == "relu":
        return nn.ReLU(inplace=inplace)
    elif name.casefold() == "lrelu":
        return nn.LeakyReLU(inplace=inplace)
    elif name.casefold() == "softmax":
        return nn.Softmax()
    elif name.casefold() == "tanh":
        return nn.Tanh()
    else:
        assert False, "Unknown activation function {}".format(name)
