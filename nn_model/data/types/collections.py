from typing import Union, Callable, List, Dict, get_args
from nn_model.data.types import InputDataPoint, InputDict


class InputDataPointList(list):
    """The InputDataPointList class is a list of InputDataPoint's"""

    def __init__(self, *inputs: InputDataPoint):
        for input in inputs:
            assert isinstance(
                input, get_args(Union[InputDataPoint, Dict])
            ), "Elements of `InputDataPointList` can only be `InputDataPoint`'s or Dict's"
        self._input_fields = (
            list(inputs[0].__annotations__.keys())
            if isinstance(inputs[0], InputDataPoint)
            else list(inputs[0].keys())
        )
        super().__init__(inputs)

    @property
    def input_fields(self):
        return self._input_fields

    def reduce(
        self,
        fields: List[str],
        func: Callable,
        *args,
        **kwargs,
    ) -> InputDataPoint:
        """The reduce function returns a single InputDataPoint object
        whose fields are going to be the result of applying a `func` to the list of elements of
        the individual InputDataPoint's
        """
        return InputDataPoint(
            **{
                field: func(
                    [
                        getattr(item, field)
                        if isinstance(item, InputDataPoint)
                        else item[field]
                        for item in self
                    ],
                    *args,
                    **kwargs,
                )
                if field in fields
                else [
                    getattr(item, field)
                    if isinstance(item, InputDataPoint)
                    else item[field]
                    for item in self
                ]
                for field in self.input_fields
            }
        )

    @classmethod
    def random_like(cls, input_data_point_list: "InputDataPointList"):
        return cls(
            *[
                InputDataPoint.random_like(input_data_point)
                for input_data_point in input_data_point_list
            ]
        )


class InputDictList(list):
    """The InputDictList class is a list of InputDict's"""

    def __init__(self, *inputs: InputDict):
        for input in inputs:
            assert isinstance(
                input, InputDict
            ), "Elements of `InputDictList` can only be `InputDict`'s"
        # Here we explicitly assume that all InputDicts have the same keys
        self._input_fields = list(inputs[0].keys())
        super().__init__(inputs)

    @property
    def input_fields(self):
        return self._input_fields

    def gather(
        self,
    ) -> InputDict:
        """The gather function returns a single InputDict object
        whose fields are going to be lists of elements of
        the individual InputDataPoint's
        """
        return InputDict(
            **{
                field: InputDataPointList(
                    *(item.get(field, None) for item in self),
                )
                for field in self.input_fields
            }
        )
