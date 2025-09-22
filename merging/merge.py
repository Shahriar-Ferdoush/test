import torch
from typing import Callable, List, Literal, Optional, Tuple, Union
from .ties_utils import TIES
from .dare_utils import DARE

MODEL_MERGERS = Literal["ties", "dare"]

MERGER_CONFIGS = {}

MERGER_CLASSES = {
    "ties": TIES,
    "dare": DARE,
}

class Merge:
    def __init__(
        self,
        merger: MODEL_MERGERS = "ties",
        config: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize a model merger.
        Args:
            merger (Literal["ties", "dare"]): Either a merger object or a string name of the merger.
                                              If a string is provided, the corresponding merger will be loaded.
                                              Supported values: "ties", "dare"
            config (Optional[dict]): Configuration dictionary for the merger.
                                    If None and a string merger name is provided, default configs will be loaded.
            **kwargs: Additional arguments to pass to the merger.
        """
        self.merger = MERGER_CLASSES[merger](config=config)

    def merge(
        self, 
        weights: List[float],                       # weights for each task vector
        base_model_parameters: torch.Tensor,        # parameters of the base model
        ft_models_parameters: List[torch.Tensor],   # list of parameters from different adapted models
        densities: List[float],                     # list of densities for trimming each task vector
        device: Optional[torch.device] = None,      # device to perform computations on. If None, uses current device
        **kwargs
    ) -> torch.Tensor:
        """
        Merge multiple models using the chosen merger.

        Args:
            weights (List[float]): Weights for each task vector.
            base_model_parameters (torch.Tensor): The parameters of the base model.
            ft_models_parameters (List[torch.Tensor]): List of parameters from different adapted models.
            densities (List[float]): List of densities for trimming each task vector.
            device (torch.device, optional): Device to perform computations on. If None, uses current device.
            **kwargs: Additional arguments to pass to the merger's merge method.
        
        Returns:
            torch.Tensor: The merged model parameters.
        """

        return self.merger.merge(
            weights=weights,
            base_model_parameters=base_model_parameters,
            ft_models_parameters=ft_models_parameters,
            densities=densities,
            device=device,
            **kwargs
        )