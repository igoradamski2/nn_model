from typing import Optional

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from nn_model.data.types import InputDataPoint
from nn_model.base import BaseModule


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, add_linear: bool = False):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

        self.add_linear = add_linear
        if add_linear:
            self.linear = nn.Sequential(nn.Linear(channels, channels), nn.Tanh())
        else:
            self.linear = nn.Sequential()

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        emb = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        if self.add_linear:
            emb = self.linear(emb)

        return emb


class SkipTransformer(nn.TransformerEncoder):
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src

        for mod in self.layers:
            new_output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

            output = output + new_output

        if self.norm is not None:
            output = self.norm(output)

        return output


class BaseTransformerEncoder(BaseModule):
    def __init__(
        self,
        transformer_layer: DictConfig,
        num_layers: int,
        add_positional_encoding: bool = True,
        positional_encoding_linear: bool = False,
        inside_skip: bool = False,
        outside_skip: bool = False,
        norm: bool = False,
    ):
        super().__init__()
        self.transformer_layer = transformer_layer
        self.num_layers = num_layers
        self.add_positional_encoding = add_positional_encoding
        self.positional_encoding_linear = positional_encoding_linear
        self.inside_skip = inside_skip
        self.outside_skip = outside_skip
        self.norm = norm

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        input_size = example_input.numeric_features.shape[-1]

        transformer_layer = instantiate(
            self.transformer_layer, d_model=input_size, batch_first=True
        )
        if self.norm:
            norm = nn.LayerNorm(input_size)
        else:
            norm = None

        if self.inside_skip:
            self.transformer = SkipTransformer(
                transformer_layer, num_layers=self.num_layers, norm=norm
            )

        else:
            self.transformer = nn.TransformerEncoder(
                transformer_layer, num_layers=self.num_layers, norm=norm
            )

        if self.add_positional_encoding:
            self.positional_encoding = PositionalEncoding1D(
                input_size, add_linear=self.positional_encoding_linear
            )

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        data = inp.numeric_features
        mask = inp.mask.clone()
        # IMPORTANT: below looks for first occurrence of False to determine sequence lengths,
        # but this doesn't work if a sequence is all true, so it is vital to get rid of these
        # incorrect input lengths in the next snippet of code.
        input_lengths = (mask.size(1) - mask.type(torch.int).argmin(dim=1)).flatten()

        # torch transformer does not work for length zero sequences, so lets remove these:
        non_zero_length_indices = (~(mask.all(dim=1))).flatten().nonzero().flatten()
        data = data.index_select(0, non_zero_length_indices)
        input_lengths = input_lengths.index_select(0, non_zero_length_indices)
        mask = mask.index_select(0, non_zero_length_indices)

        # torch positional encoding only works with sequences that are padded from the end,
        # so we have to shift our sequences accordingly (we pad from the front in algolab)
        if self.add_positional_encoding:
            shifts = (input_lengths - mask.size(1)).tolist()
            data = torch.stack(
                [torch.roll(d, shifts=s, dims=0) for d, s in zip(data, shifts)]
            )
            data = data + self.positional_encoding(data)
            # shift the padding back to the front
            data = torch.stack(
                [torch.roll(d, shifts=-s, dims=0) for d, s in zip(data, shifts)]
            )

        data = self.transformer(data, src_key_padding_mask=mask.squeeze(-1))

        # Place the non-zero-length sequences back into their original places in the batch
        final_data = torch.zeros(
            (inp.numeric_features.size(0), *data.shape[1:]),
            device=inp.numeric_features.device,
        )
        final_data.index_copy_(0, non_zero_length_indices, data)

        if self.outside_skip:
            final_data = final_data + inp.numeric_features

        inp.numeric_features = final_data
        return inp
