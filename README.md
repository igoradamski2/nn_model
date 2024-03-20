# nn_model
A simple wrapper around Pytorch that allows you to build complex models straight from a hydra config!

This software is still in a very experimental phase, so if you're interested in contributing and showing different use cases, drop me a email at igor.adamski2@gmail.com.

## Main usecase
The main usecase for this repository is for training models which have many multimodal inputs and if your experimental setup requires doing a lot of tweaks to the underlying model architecture or processing and combining data sources from different 'modes'.

Things done automatically:
- Embedding categorical variables into numeric representations
- Figuring out the shapes of registered neural networks, based on a example batch
- Automatic handling of masks
- Saving neural network configuration for easy instantiation in the future

## Tutorial
Imagine you have two inputs of different shapes:

- A 3D Timeseries of shape [100, 32, 10]
- A 1D Static vector of shape [100, 3]

In addition, the 1D static vector also includes some categorical variables, imagine there are 3 of them, each taking values 0 or 1.

You can define your example batch as follows, using the data types provided in `nn_model` (the first dimension is batch_size):
```{python}
example_batch = {
    "3d_timeseries": InputDataPoint(
        numeric_features=torch.rand(32, 100, 32, 10),
        mask=torch.zeros(32, 100, 32, 1).bool(),
    ),
    "1d_static_data": InputDataPoint(
        numeric_features=torch.rand(32, 100, 5),
        categorical_features=torch.randint(0, 2, (32, 100, 3)),
        mask=torch.zeros(32, 100, 1).bool(),
    ),
}
```

You can process such a batch with a model defined in `conf/simple_model.yaml`, given below:
```{yaml}
_target_: nn_model.model.AlgoModel
_convert_: all
operations:
  - _target_: nn_model.operations.ElementwiseOperation
    inp: [3d_timeseries]
    mod:
      - _target_: nn_model.modules.cnn.ConvWindow
        output_size: 64
        dropout: 0.2
        window: 2
        n_blocks: 2
        nonlinear: LeakyReLU
        stride: 2
        batch_norm: false
      - _target_: nn_model.modules.transformer.BaseTransformerEncoder
        _recursive_: false
        num_layers: 1
        add_positional_encoding: true
        positional_encoding_linear: false
        inside_skip: false
        outside_skip: false
        norm: true
        transformer_layer:
          _target_: torch.nn.TransformerEncoderLayer
          dropout: 0.2
          nhead: 4
          dim_feedforward: 64
          norm_first: true
    out: [3d_timeseries]
    rearrange: "batch one two feature -> (batch one) two feature"
    slice_result: "[:, :, -1, :]"

  - _target_: nn_model.operations.ElementwiseOperation
    inp: [1d_static_data]
    mod:
      - _target_: nn_model.modules.single_input.Identity
    out: [1d_static_data]

  - _target_: nn_model.operations.ReduceOperation
    inp: [3d_photo_data, 1d_static_data]
    mod: 
      - _target_: nn_model.modules.multi_input.Concatenate
        dim: -1
        fields: ["numeric_features", "mask"]
        dropout: 0
      - _target_: nn_model.modules.transformer.BaseTransformerEncoder
        _recursive_: false
        num_layers: 2
        add_positional_encoding: false
        positional_encoding_linear: false
        inside_skip: false
        outside_skip: false
        norm: true
        transformer_layer:
          _target_: torch.nn.TransformerEncoderLayer
          dropout: 0.2
          nhead: 1
          dim_feedforward: 80
          norm_first: true
    out: embedding

  - _target_: nn_model.operations.ElementwiseOperation
    inp: [embedding]
    mod:
      - _target_: nn_model.modules.single_input.FullyConnected
        output_size: 3
        add_linear: [1]
        dropout: 0.2
        nonlinear: ReLU
        add_predictor: true
        use_batch_norm: false
      - _target_: nn_model.modules.allocation.DynamicLongShortAllocator
        long_short_proportion_module: 
          _target_: nn_model.modules.single_input.Mean
          dim: -2
        allow_cash_allocation: false
    out: [allocation]

outputs: [allocation]
```

Lets walk through what the model will actually do:
1. The input `3d_timeseries` will go through a CNN and then a Transformer. The `rearrange` parameter of the first elementwise operation, tells the handler to first rearrange the input to concatenate the dimension "one" with "batch" - this means that the CNN and Transformer will act on a 3D tensor while treating the "one" dimension as constant. The `slice` paremeter, tells the handler to slice the output tensor in a given way - since we treated our 3D tensor with a transformer, we can safely slice and take the last element from dimension "two" as this is the dimension which resembles "time" in this case.
2. The input `1d_static_data` will go through a Identity module, which will do nothing to it. All that will happen is that categorical features will be turned into numeric and concatenated with them.
3. Next, the `ReduceOperation` will concatenate the features from the processed inputs `3d_timeseries` and `1d_static_data`, and will pass them through a transformer again. Since no `slice` and `rearrange` parameters have been given, the output will be the raw output after passing through the Transformer layer.
4. Finally, a FC layer will be applied to the resulting `embedding` and passed through a `DynamicLongShortAllocator` module.


In order to instantiate the model, you simply need to do this:
```{python}
from omegaconf import OmegaConf
from hydra.utils import instantiate

model_conf = OmegaConf.load("conf/simple_model.yaml")
model = instantiate(model_conf, example_batch=example_batch)
```
and the model instantiates! 

All the necessary parameters relating to the size of the networks have been figured out from the flow of the example_batch in the model - pretty cool!