from dataclasses import dataclass
from typing import List, Union

import yaml


@dataclass
class WISEHyperParams:
    # Experiments

    edit_lr: float  # Eta = Learning rate for editing
    n_iter: int
    # Method
    objective_optimization: str
    mask_ratio: float  # rouh = Random gredient mask generator ratio = 0.5 means 50% of the Wv' will not be updated
    # Wv'(i) = i-th copy of the side memory
    # Wv'(i) = Wv'(i) - eta (mask(i) * grad(Wv'(i)))              #Section 2.3.2
    alpha: float  # act_margin[0] = For irrelevant input Xi, Activation should be less than alpha
    beta: float  # act_margin[1] = For edits Xe, Activation should be greater than beta
    gamma: float  # act_margin[2] = Difference between Activation of Xe and Xi should be greater than gamma
    act_ratio: float
    merge_freq: int
    retrieve: bool  # Setting it will calculate the activation for router and choose any of the two possible paths
    replay: bool
    save_freq: Union[int, None]
    merge_alg: str
    norm_constraint: float
    # Module templates
    inner_params: List[str]
    weights: Union[float, None]
    densities: Union[float, None]

    device: str
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (
            config["merge_freq"] % config["save_freq"] == 0
        ), "merge_freq need to be divisible by save_freq (like 1000 / 500)"
        assert len(config["act_margin"]) == 3
        config["alpha"], config["beta"], config["gamma"] = (
            config["act_margin"][0],
            config["act_margin"][1],
            config["act_margin"][2],
        )
        config.pop("act_margin")

        assert (
            config and config["alg_name"] == "WISE"
        ), f'WISEHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'
        return cls(**config)
