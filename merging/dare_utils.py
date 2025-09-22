from typing import List, Optional

import torch

from .ties_utils import get_task_vector


def drop_and_rescale(
    task_vector: torch.Tensor,
    density: float,
    rescale: bool = True,
) -> torch.Tensor:
    """
    Idea behind DARE: Even if we drop more than 90% of the weights updates and
    rescale the remaining weights according to the density, the performance
    of the merged model does not degrade significantly.

    Funtionality:
        Randomly masks out elements of the tensor with probability density and
        optionally rescales the remaining elements by 1/density.

    Inputs:
        task_vector (torch.Tensor): The task vector to be trimmed.
        density (float): The density level for trimming (between 0 and 1).
        rescale (bool): Whether to rescale the remaining weights.

    Returns:
        torch.Tensor: The trimmed (and possibly rescaled) task vector.
    """

    if density >= 1.0:
        return task_vector

    if (task_vector.device.type != "cpu") or (task_vector.dtype == torch.bfloat16):
        working_dtype = task_vector.dtype
    else:
        working_dtype = torch.float32

    # Bernoulli takes a tensor of probabilities and returns a tensor
    # Value in the input range [0, 1] is treated as probability of 1 for each element
    mask = torch.bernoulli(torch.full_like(task_vector, density, dtype=working_dtype))

    result = task_vector.to(working_dtype) * mask

    if rescale and density > 0.0:
        result = result / density

    return result.to(task_vector.dtype)


class DARE:
    def __init__(self, config=None):
        if config is not None:
            self.config = config

    def merge(
        self,
        weights: List[float],  # weights for each task vector
        base_model_parameters: torch.Tensor,
        ft_models_parameters: List[torch.Tensor],
        densities: List[float],
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Merges multiple task vectors into a single parameter update using the DARE method.

        Args:
            weights (List[float]): Weights for each task vector.
            base_model_parameters (torch.Tensor): The parameters of the base model.
            ft_models_parameters (List[torch.Tensor]): List of parameters from different adapted models.
            densities (List[float]): List of densities for trimming each task vector.
            device (torch.device, optional): Device to perform computations on. If None, uses current device.

        Returns:
            torch.Tensor: The merged model parameters after applying the DARE method.
        """
        if device is None:
            device = base_model_parameters.device

        # Get task vectors
        task_vectors, base_model_parameters = get_task_vector(
            base_model_parameters,
            ft_models_parameters,
            device=device,
        )

        # Trim task vectors based on densities
        trimmed_task_vectors = [
            drop_and_rescale(tv, density)
            for tv, density in zip(task_vectors, densities)
        ]

        # Process weights
        weights = torch.tensor(
            weights,
            dtype=trimmed_task_vectors[0].dtype,
            device=device if device is not None else trimmed_task_vectors[0].device,
        )

        while len(weights.shape) < len(trimmed_task_vectors[0].shape) + 1:  # +1 for stacking dim
            weights = weights.unsqueeze(-1)


        # Weighted sum of trimmed task vectors
        weighted_task_vector = torch.stack(trimmed_task_vectors).to(weights.device) * weights
        merged_task_vector = weighted_task_vector.sum(dim=0)

        # Add merged task vector to base model parameters
        merged_model_parameters = base_model_parameters + merged_task_vector

        return merged_model_parameters


# EXAMPLE USAGE
if __name__ == "__main__":
    # Example usage of DARE merging
    base_model_params = torch.tensor([1.0, 2.0, 3.0])
    ft_model_params_1 = torch.tensor([1.5, 2.5, 3.5])
    ft_model_params_2 = torch.tensor([0.5, 1.5, 2.5])

    weights = [0.6, 0.4]
    densities = [0.8, 0.5]

    dare_merger = DARE()
    merged_params = dare_merger.merge(
        weights=weights,
        base_model_parameters=base_model_params,
        ft_models_parameters=[ft_model_params_1, ft_model_params_2],
        densities=densities,
    )

    print("Merged Model Parameters:", merged_params)
