# %%
import torch

from omegaconf import OmegaConf
from hydra.utils import instantiate

from nn_model.data.types.data import InputDataPoint
# %%
model_conf = OmegaConf.load("conf/simple_model.yaml")

# %%
# %%
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


# %%
model = instantiate(model_conf, example_batch=example_batch)
# %%
# %%
# %%
# %%