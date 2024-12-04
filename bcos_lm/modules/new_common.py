"""
Common classes and mixins for B-cos models.
"""
from typing import Union, List, Optional, Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor

__all__ = ["BcosModelBase", "ExplanationModeContextManager", "BcosSequential"]


class ExplanationModeContextManager:
    """
    A context manager which activates and puts model in to explanation
    mode and deactivates it afterwards
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.expl_modules = None

    def find_expl_modules(self):
        self.expl_modules = [
            m for m in self.model.modules() if hasattr(m, "set_explanation_mode")
        ]

    def __enter__(self):
        if self.expl_modules is None:
            self.find_expl_modules()

        for m in self.expl_modules:
            m.set_explanation_mode(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.expl_modules:
            m.set_explanation_mode(False)


class BcosModelBase(torch.nn.Module):
    """
    This base class defines useful explanation generation helpers.
    """

    to_probabilities = torch.sigmoid
    """ Function to convert outputs to probabilties. """

    def __init__(self):
        super().__init__()

        self._expl_mode_ctx = ExplanationModeContextManager(self)
        """ Context manager for explanation mode. """

    def explanation_mode(self):
        """
        Creates a context manager which puts model in to explanation
        mode and when exiting puts it into normal mode back again.
        """
        return self._expl_mode_ctx


class BcosSequential(BcosModelBase, torch.nn.Sequential):
    def __init__(self, *args):
        BcosModelBase.__init__(self)
        torch.nn.Sequential.__init__(self, *args)


class _DynamicMultiplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: "Tensor", input: "Tensor", state: "dict") -> "Tensor":
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.state = state
        ctx.save_for_backward(weight, input)
        return weight * input

    @staticmethod
    def backward(ctx, grad_output: "Tensor") -> "Tuple[Optional[Tensor], Tensor, None]":
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        weight, input = ctx.saved_tensors
        if ctx.state["fixed_weights"]:
            return None, grad_output * weight, None
        return grad_output * input, grad_output * weight, None


class DynamicMultiplication(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = {"fixed_weights": False}

    def set_explanation_mode(self, on: bool = True):
        self.state["fixed_weights"] = on

    @property
    def is_in_explanation_mode(self):
        # just for testing
        return self.state["fixed_weights"]

    def forward(self, *, weight: "Tensor", input: "Tensor") -> "Tensor":
        return _DynamicMultiplication.apply(weight, input, self.state)
    

class _DynamicMatrixMultiplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: "Tensor", input: "Tensor", state: "dict") -> "Tensor":
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.state = state
        ctx.save_for_backward(weight, input)
        return torch.matmul(weight, input)

    @staticmethod
    def backward(ctx, grad_output: "Tensor") -> "Tuple[Optional[Tensor], Tensor, None]":
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        weight, input = ctx.saved_tensors
        if ctx.state["fixed_weights"]:
            return None, torch.matmul(weight.transpose(-1, -2), grad_output), None
        return torch.matmul(grad_output, input.transpose(-1, -2)), torch.matmul(weight.transpose(-1, -2), grad_output), None


class DynamicMatrixMultiplication(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = {"fixed_weights": False}

    def set_explanation_mode(self, on: bool = True):
        self.state["fixed_weights"] = on

    @property
    def is_in_explanation_mode(self):
        # just for testing
        return self.state["fixed_weights"]

    def forward(self, *, weight: "Tensor", input: "Tensor") -> "Tensor":
        return _DynamicMatrixMultiplication.apply(weight, input, self.state)