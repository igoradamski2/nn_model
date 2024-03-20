from omegaconf import DictConfig

from dataclasses import dataclass
import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Union
from copy import deepcopy


class InputDict(dict):
    """InputDict is a special Dict of [str, InputDataPoint]"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_batch_sizes(self):
        return [val.numeric_features.shape[0] for val in self.values()]

    def get_feature_sizes(self):
        return [val.numeric_features.shape[-1] for val in self.values()]

    def select_inputs(self, keys: List[str]) -> "InputDict":
        """Selects keys from the InputDict and returns a new InputDict with cloned entries"""
        return InputDict({key: val.clone() for key, val in self.items() if key in keys})

    def clone(self) -> "InputDict":
        return InputDict({key: val.clone() for key, val in self.items()})

    def select_index(self, index: torch.Tensor) -> None:
        """Selects index from all the inputs"""
        for val in self.values():
            val.select_index(index)

    def transform(self, op: Callable, *args, **kwargs) -> "InputDict":
        return InputDict(
            {key: op(samples, *args, **kwargs) for key, samples in self.items()}
        )

    def transform_fields(self, fields: List[str], op: Callable, *args, **kwargs):
        input_dict = {}
        for key, data_point in self.items():
            for field in fields:
                field_value = getattr(data_point, field)
                if field_value is not None:
                    setattr(data_point, field, op(field_value, *args, **kwargs))
            input_dict[key] = data_point
        return InputDict(input_dict)

    def to_dict(self):
        return InputDict({key: val.to_dict() for key, val in self.items()})

    def to_input_data_point(self):
        return InputDict({key: InputDataPoint(**val) for key, val in self.items()})

    @classmethod
    def random_like(cls, input_dict: "InputDict"):
        return cls(
            {key: InputDataPoint.random_like(val) for key, val in input_dict.items()}
        )

    @classmethod
    def from_shapes_dict(
        cls, params: Dict[str, Dict[str, Union[List, Dict]]]
    ) -> "InputDict":
        return cls(
            {key: InputDataPoint.from_shapes_dict(**val) for key, val in params.items()}
        )


@dataclass
class InputDataPoint:
    """The InputDataPoint class holds all the necessary information
    about the input (numeric, categorical, categorical_map, masks etc.).

    It contains convenience methods that apply other functions
    to the elements of the InputDataPoint.
    """

    numeric_features: torch.Tensor
    mask: torch.Tensor
    categorical_features: Optional[torch.Tensor] = None
    categorical_features_map: Optional[Dict] = None
    embed_size: Optional[List] = None

    def __post_init__(self) -> None:
        """Convenience method, that will:
        - reduce lists of Nones to None
        - assert that categorical features have dimension impled by the cat_features_map
        """
        class_fields = getattr(self, "__annotations__")
        for field_name in class_fields.keys():
            field = getattr(self, field_name)
            if isinstance(field, List) and all(val is None for val in field):
                setattr(self, field_name, None)

    def remove_categoricals(self) -> None:
        self.categorical_features = None
        self.categorical_features_map = None

    def to_dict(self) -> Dict:
        class_fields = getattr(self, "__annotations__").keys()
        return {field: getattr(self, field) for field in class_fields}

    def apply(self, fields: List[str], op: Callable, *args, **kwargs) -> None:
        """Applies the op inplace to fields"""
        for field in fields:
            field_value = getattr(self, field)
            if field_value is not None:
                setattr(self, field, op(field_value, *args, **kwargs))

    def select_index(self, index: torch.Tensor) -> None:
        """Selects index from all the fields"""
        self.apply(
            ["numeric_features", "mask", "categorical_features"],
            lambda x: x[index] if x is not None else x,
        )

    def clone(self):
        return InputDataPoint(
            numeric_features=self.numeric_features.clone(),
            mask=self.mask.clone(),
            categorical_features=self.categorical_features.clone()
            if self.categorical_features is not None
            else None,
            categorical_features_map=self.categorical_features_map,
            embed_size=self.embed_size,
        )

    @property
    def shape(self) -> torch.Size:
        return self.numeric_features.shape

    @property
    def device(self) -> torch.device:
        return self.numeric_features.device

    @classmethod
    def random_like(cls, input_data_point: "InputDataPoint"):
        return cls(
            numeric_features=deepcopy(input_data_point.numeric_features.detach()),
            mask=deepcopy(input_data_point.mask),
            categorical_features=deepcopy(input_data_point.categorical_features),
            categorical_features_map=deepcopy(
                input_data_point.categorical_features_map
            ),
            embed_size=deepcopy(input_data_point.embed_size),
        )

    @classmethod
    def from_shapes_dict(
        cls,
        numeric_features: List[int],
        mask: List[int],
        categorical_features: List[int],
        categorical_features_map: Dict[str, List[str]],
        embed_size: Optional[List[int]] = None,
    ) -> "InputDataPoint":
        numeric_features = torch.rand(numeric_features)
        mask = torch.zeros(mask).bool()

        if categorical_features[-1] > 0:
            assert categorical_features[-1] == len(
                categorical_features_map.keys()
            ), "The categorical_features_map does not match the categorical_features!"

            categorical_features = torch.cat(
                [
                    torch.randint(
                        0,
                        len(categorical_features_map[key]),
                        size=categorical_features[:-1],
                    ).unsqueeze(-1)
                    for key in categorical_features_map.keys()
                ],
                dim=-1,
            )
        else:
            categorical_features = torch.empty(categorical_features, dtype=torch.int64)

        return cls(
            numeric_features=numeric_features,
            mask=mask,
            categorical_features=categorical_features,
            categorical_features_map=categorical_features_map,
            embed_size=embed_size,
        )


@dataclass
class TargetDataPoint:
    target: torch.Tensor
    benchmark: Optional[torch.Tensor] = None
    rfr: Optional[List[float]] = None

    @property
    def shape(self) -> torch.Size:
        return self.target.shape

    def clone(self) -> "TargetDataPoint":
        return TargetDataPoint(
            target=self.target.clone(),
            benchmark=self.benchmark.clone() if self.benchmark is not None else None,
            rfr=self.rfr,
        )


@dataclass
class DataPoint:
    target: TargetDataPoint
    inputs: InputDict
    key: np.ndarray
    task_name: str

    def to_dict(self) -> Dict:
        return {
            "target": self.target,
            "inputs": self.inputs.to_dict(),
            "key": self.key,
            "task_name": self.task_name,
        }

    @classmethod
    def from_datapoint(cls, datapoint: "DataPoint"):
        return cls(
            target=datapoint.target.clone(),
            inputs=datapoint.inputs.clone(),
            key=deepcopy(datapoint.key),
            task_name=datapoint.task_name,
        )
