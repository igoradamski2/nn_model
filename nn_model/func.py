from abc import abstractmethod
import torch


class BaseOutputFunc:
    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass


class Identity(BaseOutputFunc):
    def __call__(self, x: torch.Tensor):
        return x


class Slice(BaseOutputFunc):
    def __init__(self, slice: str):
        """The `slice` will essentially be a string
        and will represent a slicing operation you would
        do on a normal torch.Tensor

        so if slice="[0, 0, 0]"

        It will select the 0th element for dimensions 1,2,3
        of an N-dimensional torch.Tensor
        """
        self.slice = slice

    def __call__(self, x: torch.Tensor):
        x = eval(f"x{self.slice}")
        return x
