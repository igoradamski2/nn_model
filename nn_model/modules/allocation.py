from torch import nn
from nn_model.data.types import InputDataPoint
from nn_model.base import BaseModule
from nn_model.modules.misc import MaskedSoftmax


class LongShortAllocator(BaseModule):
    def __init__(
        self,
        long_short_proportion: float,
        allow_cash_allocation: bool = False,
    ):
        super().__init__()
        self.long_short_proportion = long_short_proportion
        self.allow_cash_allocation = allow_cash_allocation
        self.softmax = MaskedSoftmax(dim=-2)

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        pass

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        """
        predictions have to have dimension (batch, ticker, 2)
        """
        out = inp.numeric_features
        longs = (
            self.softmax(out[..., [0]], mask=inp.mask.clone())
            * self.long_short_proportion
        )
        shorts = (
            -1
            * self.softmax(out[..., [-1]], mask=inp.mask.clone())
            * (1 - self.long_short_proportion)
        )

        out = longs + shorts
        if not self.allow_cash_allocation:
            out = out / out.abs().sum(-2).unsqueeze(1)

        inp.numeric_features = out
        return inp


class DynamicLongShortAllocator(BaseModule):
    def __init__(
        self,
        long_short_proportion_module: BaseModule,
        allow_cash_allocation: bool = False,
    ):
        super().__init__()
        self.long_short_proportion = long_short_proportion_module
        self.allow_cash_allocation = allow_cash_allocation
        self.softmax = MaskedSoftmax(dim=-2)
        self.sigmoid = nn.Sigmoid()

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        self.long_short_proportion.setup(example_input)
        assert (
            example_input.numeric_features.shape[-1] == 3
        ), "DynamicLongShortAllocator requires 3 numeric features"

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        """
        predictions have to have dimension (batch, ticker, 3)
        """
        out = inp.numeric_features

        long_short_proportion_input = InputDataPoint(
            numeric_features=inp.numeric_features[..., [0]],
            mask=inp.mask,
        )

        long_short_proportion = self.long_short_proportion(long_short_proportion_input)
        long_short_proportion = self.sigmoid(
            long_short_proportion.numeric_features
        ).squeeze()
        long_short_proportion = (
            long_short_proportion.unsqueeze(-1).unsqueeze(-1).expand_as(out[..., [1]])
        )
        longs = self.softmax(out[..., [1]], mask=inp.mask) * long_short_proportion
        shorts = (
            -1
            * self.softmax(out[..., [-1]], mask=inp.mask)
            * (1 - long_short_proportion)
        )

        out = longs + shorts
        if not self.allow_cash_allocation:
            out = out / (out.abs().sum(-2).unsqueeze(1) + 1e-9)

        inp.numeric_features = out
        return inp
