from copy import deepcopy
from typing import Sequence

import torch

from nn_model.data.types import InputDataPoint, InputDict, InputDataPointList
from nn_model.base import BaseModule, BaseOperation
from loguru import logger


class ElementwiseOperation(BaseOperation):
    """A operation class that applies a module element-wise to each input"""

    def setup_pass(
        self,
        input_dict: InputDict,
    ) -> None:
        """Here setup will take the operation (module)
        and instantiate it PER EACH input
        """
        assert isinstance(
            self.out, Sequence
        ), "ElementwiseOperation assumes that out is a list"

        assert len(self.inp) == len(
            self.out
        ), "ElementwiseOperation assumes that we get the same amount of outputs as inputs"

        module_dict = torch.nn.ModuleDict({})
        for input_name in self.inp:
            # Extract the example input for the input_name in question
            example_input: InputDataPoint = input_dict[input_name]
            # example_extra_inp = (
            #     input_dict.select_inputs(self.extra_inp)
            #     if self.extra_inp is not None
            #     else None
            # )

            # Deepcopy the main module and set it up with the example input
            module_dict[input_name] = torch.nn.ModuleList([])
            for mod in self.module:
                module: BaseModule = deepcopy(mod)

                # Setup the individual modules
                example_input = module(example_input)

                # Update the module dict
                module_dict[input_name].append(module)

        self.module = module_dict

    def forward_pass(self, inputs: InputDict) -> InputDict:
        # extra_inp = inputs.select_inputs(self.extra_inp)
        outputs = InputDict({})
        for input_name, output_name in zip(self.inp, self.out):
            for mod in self.module[input_name]:
                out = mod(inputs[input_name])  # extra_inp=extra_inp)

            outputs[output_name] = out
        return outputs


class ReduceOperation(BaseOperation):
    """A operation class that applies a module to all inputs at once, reducing it to a single output"""

    def setup_pass(
        self,
        input_dict: InputDict,
    ):
        """Here setup will take the operation (module)
        and instantiate one FOR ALL INPUTS
        """
        assert isinstance(
            self.out, str
        ), "ReduceOperation assumes that we get just 1 output (str)"

        example_input = InputDataPointList(
            *(input_dict[input_name] for input_name in self.inp),
        )

        # Setup the main module with the example input
        module_list = torch.nn.ModuleList([])
        for module in self.module:
            # Setup the individual modules
            example_input = module(example_input)
            # Update the module list
            module_list.append(module)

        self.module = module_list

    def forward_pass(self, inputs: InputDict) -> InputDict:
        inp = InputDataPointList(
            *(inputs[input_name] for input_name in self.inp),
        )
        for mod in self.module:
            inp = mod(inp)

        return InputDict({self.out: inp})
