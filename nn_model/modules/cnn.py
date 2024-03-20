from torch import nn, Tensor
from nn_model.data.types.data import InputDataPoint

from nn_model.base import BaseModule


class ConvWindow(BaseModule):
    def __init__(
        self,
        output_size: int,
        dropout: float,
        window: int,
        n_blocks: int,
        nonlinear: str,
        stride: int,
        batch_norm: bool,
    ):
        super().__init__()
        self.output_size = output_size
        self.dropout = dropout
        self.window = window
        self.n_blocks = n_blocks
        self.nonlinear = nonlinear
        self.stride = stride
        self.batch_norm = batch_norm

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        assert (
            len(example_input.numeric_features.shape) == 3
        ), "ConvWindow expects 3 dimensions only"

        input_size = example_input.numeric_features.shape[-1]

        self.dropout = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList(
            [
                self.conv_block(
                    n,
                    self.window,
                    input_size,
                    self.output_size,
                    self.nonlinear,
                    self.stride,
                    self.batch_norm,
                )
                for n in range(self.n_blocks)
            ]
        )

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        data = inp.numeric_features
        data = self.dropout(data)
        data = data.transpose(1, 2)

        mask = inp.mask

        for block in self.blocks:
            data = self.pad(data, block)
            data = block(data)
            data = self.dropout(data)

        data = data.transpose(1, 2)
        inp.numeric_features = data
        # Now that we have made the forward pass, we must get rid of the padding in
        # the sequence dimension:
        mask[~mask.all(dim=1).squeeze(-1), ...] = False
        mask = mask[:, : data.shape[1], :]
        inp.mask = mask
        return inp

    def pad(self, data: Tensor, block: nn.Module):
        data_size = data.size(2)
        kernel_size = block.conv.kernel_size[0]
        if data_size < kernel_size:
            pad_size = kernel_size - data_size
        elif data_size == kernel_size:
            return data
        else:
            stride = block.conv.stride[0]
            stride_modulo = (data_size - kernel_size) % stride
            if stride_modulo == 0:
                return data
            else:
                pad_size = stride - stride_modulo
        return nn.functional.pad(data, (pad_size, 0), mode="constant", value=0)

    def conv_block(
        self,
        n: int,
        window: int,
        emb_size: int,
        channels: int,
        nonlinear: str,
        stride: int,
        batch_norm: bool,
    ) -> nn.Module:
        layer = nn.Sequential()
        if n > 0:
            emb_size = channels
        layer.add_module(
            "conv",
            nn.Conv1d(
                in_channels=emb_size,
                out_channels=channels,
                kernel_size=window,
                stride=stride,
            ),
        )
        if batch_norm:
            layer.add_module("norm", nn.BatchNorm1d(num_features=channels))
        layer.add_module("nonlinear", getattr(nn, nonlinear)())
        return layer
