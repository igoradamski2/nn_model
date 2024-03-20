from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from omegaconf import DictConfig

import torch
from torch import nn

from einops import rearrange

from nn_model.data.types import InputDataPoint, InputDict
from nn_model.data.types.collections import InputDataPointList
from nn_model.func import Slice

from loguru import logger


class BaseModelElement(ABC, nn.Module):
    """A base model element, which serves as a basic
    building block for Modules and Operations

    It handles the logic of initialization (the fact that you cant
    initialize a module twice)
    """

    def __init__(self):
        super().__init__()

        self.initialized = False

    @abstractmethod
    def forward_pass(
        self, inp: Union[InputDataPoint, InputDataPointList]
    ) -> InputDataPoint:
        pass

    @abstractmethod
    def setup_pass(self, example_input: InputDataPoint) -> None:
        pass

    @abstractmethod
    def setup(self, example_input: Union[InputDataPoint, InputDict]) -> None:
        pass

    @abstractmethod
    def forward(
        self, inp: Union[InputDataPoint, InputDict]
    ) -> Union[InputDataPoint, InputDict]:
        pass

    @property
    def initialized(self) -> bool:
        return self._initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            raise ValueError("Cannot initialize a BaseModule twice!!")
        self._initialized = value


# ================================================================================


class BaseModule(BaseModelElement):
    """A base class for a module, that holds BaseNeuralModule's

    A BaseModule always takes in an example input and
    build the module from that

    The children of this class should implement
        forward_pass
        setup_pass
    """

    @abstractmethod
    def setup_pass(self, example_input: InputDataPoint) -> None:
        """This method instantiates all the modules necessary to run the operation"""
        pass

    @abstractmethod
    def forward_pass(
        self,
        example_input: InputDataPoint,
    ) -> InputDict:
        """This method runs the operation on the input"""
        pass

    def setup(self, example_input: InputDataPoint) -> None:
        self.initialize_emb_layers(
            example_input.categorical_features_map, example_input.embed_size
        )
        self.setup_pass(self.add_categorical_to_numeric(example_input))
        self.initialized = True

    def add_categorical_to_numeric(self, inp: InputDataPoint) -> InputDataPoint:
        if inp.categorical_features_map is not None:
            embs = torch.jit.annotate(List[torch.Tensor], [])
            for i, (_, emb_layer) in enumerate(self.emb_layers.items()):
                if inp.categorical_features.shape[-1] > 0:
                    emb = emb_layer(inp.categorical_features[..., i].long())
                else:
                    emb = (
                        torch.zeros(
                            size=inp.numeric_features.shape[:-1]
                            + (emb_layer.embedding_dim,)
                        )
                        .float()
                        .cuda()
                    )
                # Here we need to manually set the masked parts of the embedding to 0
                emb = emb * ((~inp.mask.expand_as(emb)).float())
                embs.append(emb)
            numeric_input = torch.cat(embs + [inp.numeric_features], dim=-1)
        else:
            numeric_input = inp.numeric_features

        inp.numeric_features = numeric_input
        return inp

    def initialize_emb_layers(
        self, cat_feat_map: Dict, embed_size: Optional[List] = None
    ) -> int:
        self.emb_layers = nn.ModuleDict({})
        embs_feat_dim = 0
        if cat_feat_map is not None:
            for i, feature in enumerate(cat_feat_map):
                emb_dim = (
                    self.compute_emb_dim(len(cat_feat_map[feature]))
                    if embed_size is None
                    else embed_size[i]
                )
                embs_feat_dim += emb_dim
                self.emb_layers[feature] = nn.Embedding(
                    num_embeddings=len(cat_feat_map[feature]), embedding_dim=emb_dim
                )

    def compute_emb_dim(self, emb_size: int) -> int:
        """TODO:
        there should be a way to store information about a way we compute embeddings size in exp results
        """
        return min(max(int(emb_size / 2), 10), 50)

    def forward(self, inp: InputDataPoint) -> InputDataPoint:
        if not self.initialized:
            self.setup(inp.random_like(inp))

        inp = self.add_categorical_to_numeric(inp)
        inp.remove_categoricals()
        return self.forward_pass(inp)


class BaseOperation(BaseModelElement):
    def __init__(
        self,
        inp: List[str],
        mod: List[BaseModule],
        out: Union[List[str], str],
        extra_inp: List[str] = None,
        rearrange: str = None,
        slice_result: str = None,
    ) -> None:
        """A BaseOperation class that handles applying operations to inputs

        - inp is a list of strings that represent the input names

        - mod is a list of BaseModules that will be applied to the input sequentially

        - out is a list of strings that represent the output names

        - extra_inp will be passed as extra input to the module during forward

        - rearrange is a string that represents the rearrange pattern, which will be
        applied to the input before passing it to the module and then the inverse
        of the rearrangement will be applied to the output

        - slice_result is a string that represents a slice that will be applied to the
        output at the end of passing it through the module and the rearrangement
        """
        super().__init__()
        self.inp = inp
        self.out = out
        self.module = mod

        self.extra_inp = extra_inp if extra_inp is not None else []
        if rearrange is not None:
            assert "batch" in rearrange, (
                "The rearrange pattern must have the word 'batch' - "
                "its always the first dimension"
            )
            assert "feature" in rearrange, (
                "The rearrange pattern must have the word 'feature' - "
                "its always the last dimension"
            )
            assert "->" in rearrange, (
                "The rearrange pattern must have the pattern '->' - "
                "its represents the delimiter between the input and output dimensions"
            )

        self.rearrange = rearrange
        self.slice_result = Slice(slice_result) if slice_result is not None else None

    @abstractmethod
    def setup_pass(self, input_dict: InputDict) -> None:
        """This method instantiates all the modules necessary to run the operation"""
        pass

    @abstractmethod
    def forward_pass(
        self,
        inputs: InputDict,
    ) -> InputDict:
        """This method runs the operation on the input"""
        pass

    def setup(self, input_dict: InputDict) -> InputDict:
        input_dict = self.setup_pass(input_dict)
        self.initialized = True

    def rearrange_forwards(self, inputs: InputDict) -> InputDict:
        return inputs.transform_fields(
            fields=["numeric_features", "categorical_features", "mask"],
            op=rearrange,
            pattern=self.rearrange,
        )

    def rearrange_backwards(
        self, inputs: InputDict, batch_sizes: List[int], feature_shapes: List[int]
    ) -> InputDict:
        backwards_pattern = " -> ".join(self.rearrange.split(" -> ")[::-1])
        for (key, inp), batch_size, feature_shape in zip(
            inputs.items(), batch_sizes, feature_shapes
        ):
            inp.apply(
                fields=["numeric_features", "categorical_features"],
                op=rearrange,
                pattern=backwards_pattern,
                batch=batch_size,
                feature=feature_shape,
            )

            inp.apply(
                fields=["mask"],
                op=rearrange,
                pattern=backwards_pattern,
                batch=batch_size,
                feature=1,
            )

            inputs[key] = inp

        return inputs

    def slice(
        self,
        inputs: InputDict,
        slice_result: Slice,
    ):
        return inputs.transform_fields(
            fields=["numeric_features", "mask"],
            op=slice_result,
        )

    def forward(
        self,
        inputs: InputDict,
    ):
        if self.rearrange is not None:
            batch_sizes = inputs.get_batch_sizes()
            inputs = self.rearrange_forwards(inputs)

        if not self.initialized:
            self.setup(InputDict.random_like(inputs))

        out: InputDict = self.forward_pass(inputs)

        if self.rearrange is not None:
            out = self.rearrange_backwards(
                out, batch_sizes=batch_sizes, feature_shapes=out.get_feature_sizes()
            )

        if self.slice_result is not None:
            out = self.slice(out, self.slice_result)

        return out
