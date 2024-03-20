from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nn_model.data.types import InputDataPoint
from nn_model.base import BaseModule


class RNN(BaseModule):
    def __init__(
        self,
        rnn_layer: DictConfig,
    ):
        super().__init__()
        self.rnn_layer = rnn_layer

    def setup_pass(
        self,
        example_input: InputDataPoint,
    ):
        # TODO: Separate the RNN parameters, so that if we dont pass hidden_size it just keeps
        # it as the input size, or does like in Linear
        input_size = example_input.numeric_features.shape[-1]
        self.rnn_layer = instantiate(
            self.rnn_layer, input_size=input_size, batch_first=True
        )

    def forward_pass(self, inp: InputDataPoint) -> InputDataPoint:
        data = inp.numeric_features
        mask = inp.mask.clone()
        # IMPORTANT: below looks for first occurrence of False to determine sequence lengths,
        # but this doesn't work if a sequence is all true, so it is vital to get rid of these
        # incorrect input lengths in the next snippet of code.
        input_lengths = (mask.size(1) - mask.type(torch.int).argmin(dim=1)).flatten()

        # torch sequence packing does not work for length zero sequences, so lets remove these:
        non_zero_length_indices = (~(mask.all(dim=1))).flatten().nonzero().flatten()
        data = data.index_select(0, non_zero_length_indices)
        input_lengths = input_lengths.index_select(0, non_zero_length_indices)

        # torch sequence packing also only works with sequences that re padded from the end,
        # so we have to shift our sequences accordingly (we pad from the front in algolab)
        shifts = (input_lengths - mask.size(1)).tolist()
        data = torch.stack(
            [torch.roll(d, shifts=s, dims=0) for d, s in zip(data, shifts)]
        )
        # Pack the sequences
        packed_data = pack_padded_sequence(
            data, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # forward pass and pad the packed output
        packed_data, _ = self.rnn_layer(packed_data)
        data, _ = pad_packed_sequence(packed_data, batch_first=True)
        # shift the padding back to the front
        data = torch.stack(
            [torch.roll(d, shifts=-s, dims=0) for d, s in zip(data, shifts)]
        )
        # Place the non-zero-length sequences back into their original places in the batch
        final_data = torch.zeros(
            (inp.numeric_features.size(0), *data.shape[1:]),
            device=inp.numeric_features.device,
        )
        final_data.index_copy_(0, non_zero_length_indices, data)
        inp.numeric_features = final_data
        return inp
