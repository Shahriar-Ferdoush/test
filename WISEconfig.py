from dataclasses import dataclass
from typing import List, Union
import yaml

@dataclass
class WISEConfig:
    edit_lr: float                  # Eta: Learning rate for editing
    n_iter: int                     # Number of iterations for editing
    num_train_steps: int            # repeat of n_iter for training the side memory

    # Method
    objective_optimization: str     # "only_label"
    mask_ratio: float               # ro = Random gredient mask generator reatio = 0.5 means 50% of Wv' will be updated
                                    # Wv'(i) = i-th copy of the side memory
                                    # Wv'(i) = Wv'(i) - eta (mask(i) * grad(Wv'(i)))              #Section 2.3.2
    alpha: float                    # act_margin[0] = For irrelevant input Xi, Activation should be less than alpha
    beta: float                     # act_margin[1] = For edits input Xe, Activation should be more than beta
    gamma: float                    # act_margin[2] = Difference between Activation for Xe and Xi should be greater than gamma

    activation_ratio: float         # Rescale minimum threshold for memory selection

    merge_freq: int
    retrieve: bool                  # True for choose any path from the router for inferencing
    replay: bool

    save_freq: Union[int, None]     # 
    merge_alg: str                  # Which algorithm to choose for side memories merging
    
    norm_contrain: float


    # Module
    inner_params: List[str]         # List of parameters to choose for side memory
    weights: Union[float, None]     # NOT SURE: Weight to put on edits to be merged 
    densities: Union[float, None]   # NOT SURE: Density to put on edits to be merged



    # Model - But not needed right now
    device: int                     # Device to run the model on
    alg_name: str                   # Algorithm name: "WISE"
    model_name: str                 # Model name: "gpt2-medium"

    batch_size: int = 1             # Batch size for editing
    max_lenght: int = 30
    model_parallel: bool = False



