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

from .common import DetachableModule, DynamicMultiplication

__all__ = ["NormedLinear", "BcosLinear", "BcosGELUActivation"]


class NormedLinear(nn.Linear):
    """
    Standard linear transform, but with unit norm weights.
    """

    def forward(self, input: Tensor) -> Tensor:
        w = self.weight / LA.vector_norm(self.weight, dim=1, keepdim=True)
        return F.linear(input, w, self.bias)


class BcosLinear(DetachableModule):
    """
    BcosLinear is a linear transform with unit norm weights and a cosine similarity
    activation function. The cosine similarity is calculated between the input
    vector and the weight vector. The output is then scaled by the cosine
    similarity.

    See the paper for more details: https://arxiv.org/abs/2205.10268

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        This is ignored. BcosLinear does not support bias.
    device : Optional[torch.device]
        The device of the weights.
    dtype : Optional[torch.dtype]
        The dtype of the weights.
    b : int | float
        The base of the exponential used to scale the cosine similarity.
    max_out : int
        The number of output vectors to use. If this is greater than 1, the
        output is calculated as the maximum of `max_out` vectors. This is
        equivalent to using a MaxOut activation function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        b: Union[int, float] = 2,
        max_out: int = 1,
    ) -> None:
        assert not bias
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = False

        self.b = b
        self.max_out = max_out

        self.linear = NormedLinear(
            in_features,
            out_features * self.max_out,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (*, H_in)

        Returns:
            B-cos Linear output on the input tensor.
            Shape: (*, H_out)
        """
        # Simple linear layer
        out = self.linear(in_tensor)

        # MaxOut computation
        if self.max_out > 1:
            M = self.max_out
            O = self.out_features  # noqa: E741
            out = out.unflatten(dim=-1, sizes=(O, M))
            out = out.max(dim=-1, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1:
            return out

        # Calculating the norm of input vectors ||x||
        norm = LA.vector_norm(in_tensor, dim=-1, keepdim=True) + 1e-12

        # Calculate the dynamic scale (|cos|^(B-1))
        # Note that cos = (x·ŵ)/||x||
        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm = norm.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / norm
        else:
            abs_cos = (maybe_detached_out / norm).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        # put everything together
        out = dynamic_scaling * out  # |cos|^(B-1) (ŵ·x)
        return out

    def extra_repr(self) -> str:
        # rest in self.linear
        s = "B={b}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        # final comma as self.linear is shown in next line
        s += ","

        return s.format(**self.__dict__)
    
class BcosGELUActivation(DetachableModule):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()     
        self.act = self._gelu_python


    def _gelu_python(self, input: Tensor) -> Tensor:
        dynamic_scaling = 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        if self.detach:
            dynamic_scaling = dynamic_scaling.detach()
        return input * dynamic_scaling

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)