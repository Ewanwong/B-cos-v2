"""
Contains a Linear layer which uses the B-cos transform.

NOTE: In case you're wondering why the convolution models do not use
`BcosLinear`, it's because maintaining two versions of essentially
the same thing would be very error-prone during development and testing!
"""
from typing import Union
import math
import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .new_common import BcosModelBase, DynamicMultiplication

__all__ = ["NormedLinear", "BcosLinear", "BcosGELUActivation"]


class NormedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, bias=False)

    def forward(self, in_tensor: Tensor) -> Tensor:
        w = self.weight / self.weight.norm(p=2, dim=1, keepdim=True)
        return F.linear(in_tensor, w, self.bias)


class BcosLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        # bcos specific
        b: Union[int, float] = 2,
        max_out: int = 1,
        normalize_weights: bool = True,
        **kwargs,
    ):
        assert max_out > 0, f"max_out should be greater than 0, was {max_out}"
        super().__init__()
        self.linear = NormedLinear(in_features, out_features * max_out)

        self.in_features = in_features
        self.out_features = out_features
        self.b = b
        self.max_out = max_out
        self.normalized_weights = normalize_weights
        self.dynamic_multiplication = DynamicMultiplication()

    def forward(self, in_tensor: Tensor) -> Tensor:
        out = self.linear(in_tensor)

        # max out computation
        if self.max_out > 1:
            M = self.max_out
            D = self.out_features
            out = out.unflatten(dim=-1, sizes=(D, M))
            out = out.max(dim=-1, keepdim=False).values

        if self.b == 1:  # no need to go further
            return out

        norm = (in_tensor ** 2).sum(dim=-1, keepdim=True).add(1e-6).sqrt()

        # add weight norm if weights are unnormalized
        if not self.normalized_weights:
            w = self.linear.weight
            norm = norm * w.norm(p=2, dim=1)

        # b = 2 allows for faster version
        if self.b == 2:
            dynamic_weights = out.abs() / norm
        else:
            abs_cos = (out / norm).abs() + 1e-6  # |cos| term
            dynamic_weights = abs_cos.pow(self.b - 1)

        out = self.dynamic_multiplication(weight=dynamic_weights, input=out)
        return out

    def extra_repr(self) -> str:
        # rest in self.projection
        s = "B={b}, normalized_weights={normalized_weights}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        # final comma as self.linear is shown in next line
        s += ","

        return s.format(**self.__dict__)
    
class BcosGELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()     
        self.act = self._gelu_python
        self.dynamic_multiplication = DynamicMultiplication()


    def _gelu_python(self, input: Tensor) -> Tensor:
        dynamic_scaling = 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        output = self.dynamic_multiplication(weight=dynamic_scaling, input=input)
        return input * dynamic_scaling

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)