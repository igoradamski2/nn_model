import torch
from torch import nn


class MaskedSoftmax(nn.Softmax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = (mask.long() * -1e30).expand_as(input)

        input = input + mask
        return super().forward(input)


class BatchNorm1dSqUnsq(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.batch_norm_layer = nn.BatchNorm1d(input_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
            x = self.batch_norm_layer(x)
            x = x.unsqueeze(1)
        else:
            x = self.batch_norm_layer(x)

        return x
