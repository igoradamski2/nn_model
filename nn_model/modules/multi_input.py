from typing import List
import torch

from nn_model.data.types import InputDataPointList
from nn_model.data.types.data import InputDataPoint
from nn_model.base import BaseModule


class Concatenate(BaseModule):
    def __init__(
        self,
        dim: int,
        fields: List[str],
        dropout: float = None,
    ):
        super().__init__()
        self.dim = dim
        self.fields = fields
        self.dropout = dropout

    def setup(self, example_input: InputDataPointList) -> None:
        self.setup_pass(example_input)
        self.initialized = True

    def setup_pass(
        self,
        example_input: InputDataPointList,
    ):
        pass

    def inputs_dropout(self, inp: InputDataPointList) -> InputDataPointList:
        if len(inp) == 1:
            # If there is only one input it does not make sense to apply dropout
            return inp

        if self.dropout is not None and self.training:
            if torch.rand(1) < self.dropout:
                k_drop = torch.randperm(len(inp) - 1)[0] + 1
                features_to_drop = torch.randperm(len(inp))[:k_drop]
                for i in features_to_drop:
                    inp[i.item()].apply(
                        fields=["numeric_features"],
                        op=torch.nn.functional.dropout,
                        p=1.0,
                    )

        return inp

    def forward_pass(self, inp: InputDataPointList) -> InputDataPoint:
        inp = self.inputs_dropout(inp)

        out = inp.reduce(
            fields=self.fields,
            func=torch.cat,
            dim=self.dim,
        )
        # Compute the union of masks of the inputs that we are concatenating
        out.mask = out.mask.any(dim=-1).unsqueeze(-1)

        return out

    def forward(self, inp: InputDataPointList) -> InputDataPoint:
        if not self.initialized:
            self.setup(inp.random_like(inp))

        for i, _inp in enumerate(inp):
            _inp = self.add_categorical_to_numeric(_inp)
            _inp.remove_categoricals()
            inp[i] = _inp

        return self.forward_pass(inp)
