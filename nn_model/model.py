from copy import deepcopy
from typing import Dict, List, Union
from torch import nn
import torch

from nn_model.data.types import InputDict, InputDataPoint
from nn_model.base import BaseOperation

from loguru import logger


class AlgoModel(nn.Module):
    def __init__(
        self,
        operations: List[BaseOperation],
        outputs: List[str],
        example_batch: Dict[str, InputDict] = None,
        net_params: Dict[str, Dict[str, Union[List, Dict]]] = None,
    ):
        """
        operate_inplace: if True, the operations will be applied in-place
            if you have any operations that reuse previous outputs, you should
            be careful and set this to False. When this value is True, the
            model is much more memory efficient.
        """
        super().__init__()

        if example_batch is None and net_params is None:
            raise ValueError(
                "Either example_batch or net_params must be provided to AlgoModel"
            )

        if example_batch is None:
            example_batch = InputDict.from_shapes_dict(net_params)
        else:
            example_batch = InputDict(example_batch).to_input_data_point()

        self._net_params = self.extract_net_params(inputs=example_batch)
        self.outputs = outputs

        self.operations = nn.ModuleList(modules=operations)

        # Make a forward pass through the model to initialize it
        example_output = self.forward(example_batch)
        self.output_dim_dict = {
            key: value.shape[-1] for key, value in example_output.items()
        }

        for output in outputs:
            assert (
                output in example_output.keys()
            ), f"The requested output={output} has not been created by the model!"

    @property
    def net_params(self):
        return self._net_params

    @staticmethod
    def extract_net_params(inputs: InputDict[str, InputDataPoint]):
        net_params = {}
        for key, value in inputs.items():
            net_params[key] = {}
            for field_name in value.__annotations__.keys():
                field_value = getattr(value, field_name)
                if isinstance(field_value, torch.Tensor):
                    net_params[key][field_name] = list(field_value.shape)
                else:
                    net_params[key][field_name] = deepcopy(field_value)
        return net_params

    def forward(self, inputs: InputDict) -> Dict[str, InputDataPoint]:
        for operation in self.operations:
            # Pick only the inputs needed for that Operation
            curr_inputs = inputs.select_inputs(operation.inp + operation.extra_inp)
            curr_outputs = operation(curr_inputs)

            # Update the input dict with new outputs
            inputs = InputDict(inputs | curr_outputs)

        return {key: inputs[key] for key in self.outputs}
