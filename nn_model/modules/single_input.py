from typing import List
from einops import rearrange
import torch.nn as nn
import torch

from nn_model.data.types import InputDataPoint, InputDataPointList, InputDict
from nn_model.base import BaseModule
from nn_model.modules.misc import BatchNorm1dSqUnsq


class Identity(BaseModule):
    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        pass

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        return inp


class Mean(BaseModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        pass

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        # We need to aviod taking mean of elements that are masked out
        inp.numeric_features = (inp.numeric_features * (~inp.mask)).sum(self.dim) / (
            (~inp.mask).sum(dim=self.dim) + 1e-12
        )
        return inp


class Tanh(BaseModule):
    def __init__(self, const: float = 1.0, hard: bool = False):
        super().__init__()
        self.const = const
        if hard:
            self.tanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        else:
            self.tanh = nn.Tanh()

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        pass

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        inp.numeric_features = torch.mul(self.tanh(inp.numeric_features), self.const)
        return inp


class InputRearrange(BaseModule):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        pass

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        for field in ["numeric_features", "mask"]:
            field_value = getattr(inp, field)
            if field_value is not None:
                setattr(inp, field, rearrange(field_value, self.pattern))
        return inp


class FullyConnected(BaseModule):
    def __init__(
        self,
        output_size: int,
        add_linear: list,
        dropout: float,
        nonlinear: str,
        use_batch_norm: bool = False,
        add_predictor: bool = False,
    ):
        super().__init__()
        self.output_size = output_size
        self.add_linear = add_linear
        self.dropout = dropout
        self.nonlinear = nonlinear
        self.use_batch_norm = use_batch_norm
        self.add_predictor = add_predictor

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        input_size = example_input.numeric_features.shape[-1]
        linear_layers = []
        for al in self.add_linear:
            hidden_size = int(input_size * al)
            linear_layers.append(nn.Dropout(self.dropout))
            linear_layers.append(nn.Linear(input_size, hidden_size))
            if self.use_batch_norm:
                linear_layers.append(BatchNorm1dSqUnsq(hidden_size))
            if self.nonlinear is not None:
                linear_layers.append(getattr(nn, self.nonlinear)())
            input_size = hidden_size

        if self.add_predictor:
            predictor = nn.Sequential(
                nn.Dropout(self.dropout), nn.Linear(input_size, self.output_size)
            )
        else:
            predictor = nn.Sequential()
            linear_layers.append(nn.Dropout(self.dropout))
            linear_layers.append(nn.Linear(input_size, self.output_size))
            if self.use_batch_norm:
                linear_layers.append(BatchNorm1dSqUnsq(self.output_size))
            if self.nonlinear is not None:
                linear_layers.append(getattr(nn, self.nonlinear)())

        self.module = nn.Sequential(*linear_layers, predictor)

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        data = inp.numeric_features
        inp.numeric_features = self.module(data)
        return inp


class Sort(BaseModule):
    # TODO: This has not been confirmed to work
    def __init__(
        self,
        by_input: str,
        dim: int,
        fields: List[str] = None,
        by_field: str = "numeric_features",
    ):
        """Apply sorting

        by_input: str -> input by whcih to sort
        """
        super().__init__()
        self.by_input = by_input
        self.dim = dim
        self.fields = fields
        self.by_field = by_field

        raise NotImplementedError("`Sort` needs to be checked")

    def setup(
        self,
        example_input: InputDataPointList,
    ):
        pass

    def forward(self, inp: InputDataPoint, extra_inp: InputDict) -> InputDataPoint:
        indices = torch.argsort(
            getattr(extra_inp[self.by_input], self.by_field),
            dim=self.dim,
            descending=True,
            stable=True,
        )

        inp.apply(
            torch.gather,
            dim=self.dim,
            index=indices.expand_as(inp.numeric_features),
        )
        return inp
