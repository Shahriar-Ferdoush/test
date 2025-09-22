from typing import List, Literal, Optional, Tuple

import torch


def get_task_vector(
    base_model_parameters: torch.Tensor,
    ft_models_parameters: List[torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Computes the task vector as the difference between the adapted model parameters
    and the base model parameters.

    Args:
        base_model_parameters (torch.Tensor): The parameters of the base model.
        ft_models_parameters (List[torch.Tensor]): List of parameters from different fine-tuned models.
        device (torch.device, optional): Device to perform computations on. If None, uses current device.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]:
            - List of parameter differences (task vectors) for each fine-tuned model.
            - Base model parameters.
    """
    # Move base parameters to specified device if provided
    if device is not None:
        base_model_parameters = base_model_parameters.to(device)
        ft_models_parameters = [params.to(device) for params in ft_models_parameters]

    # Stack fine-tuned model parameters for vectorized computation
    ft_params_stacked = torch.stack(ft_models_parameters)
    task_vectors_stacked = ft_params_stacked - base_model_parameters.unsqueeze(0)

    # Convert task vectors tensor to list of tensors
    task_vectors = list(task_vectors_stacked.unbind(dim=0))

    return task_vectors, base_model_parameters


def trim(
    task_vector: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """
    Trims the task vector to retain only the top-k% of its elements based on absolute value.

    Args:
        task_vector (torch.Tensor): The task vector to be trimmed.
        density (float): The fraction of elements to retain (between 0 and 1).

    Returns:
        torch.Tensor: The trimmed task vector with only the top-k% elements retained.
    """
    if density >= 1.0:
        return task_vector

    # Calculate the number of elements to retain
    k = int(density * task_vector.numel())
    if k < 0:
        raise ValueError("Density must be between 0 and 1.")

    # Get the threshold value for trimming
    threshold = torch.topk(task_vector.abs().view(-1), k).values.min()

    # Create a mask for elements to retain
    mask = task_vector.abs() >= threshold
    return task_vector * mask


def get_elect_mask(
    task_vectors_stacked: torch.Tensor,
    method: Literal["sum", "count"] = "sum",  # TIES-merging uses "sum"
    mask_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Computes a boolean mask indicating where each task vector's sign matches the elected sign.

    Args:
        task_vectors_stacked (torch.Tensor): Stacked task vectors of shape (num_tasks, num_params).
        method (Literal["sum", "count"]): Method to compute the sign mask. "sum" sums the task vectors,
                                          "count" counts the number of positive and negative signs.
        mask_dtype (torch.dtype, optional): Desired data type of the output mask. If None, uses the same dtype as input.

    Returns:
        torch.Tensor: A boolean mask of shape (num_tasks, num_params) indicating where each task vector's sign matches the elected sign.
    """
    if mask_dtype is None:
        mask_dtype = task_vectors_stacked.dtype

    sign = task_vectors_stacked.sign().to(mask_dtype)

    if method == "sum":
        elected_sign = (task_vectors_stacked.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    elif method == "count":
        elected_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sum' or 'count'.")

    return sign == elected_sign  # True where sign matches elected sign





# FOR TIES MERGING
# TESTING CODE AT THE BOTTOM
class TIES:
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
        Merges multiple task vectors into a single parameter update using the TIES method.

        Args:
            weights (List[float]): Weights for each task vector.
            base_model_parameters (torch.Tensor): The parameters of the base model.
            ft_models_parameters (List[torch.Tensor]): List of parameters from different adapted models.
            densities (List[float]): List of densities for trimming each task vector.
            device (torch.device, optional): Device to perform computations on. If None, uses current device.

        Returns:
            torch.Tensor: The merged model parameters after applying the TIES method.
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
            trim(tv, density) for tv, density in zip(task_vectors, densities)
        ]

        # Stack trimmed task vectors for vectorized operations
        task_vectors_stacked = torch.stack(trimmed_task_vectors).to(device)

        # Compute the election mask
        elect_mask = get_elect_mask(task_vectors_stacked, method="sum").to(device if device is not None else task_vectors_stacked.device)

        # Process weights
        weights = torch.tensor(
            weights,
            dtype=task_vectors_stacked.dtype,
            device=device if device is not None else task_vectors_stacked.device,
        )

        while len(weights.shape) < len(task_vectors_stacked.shape):
            weights = weights.unsqueeze(-1)

        # Apply weights and election mask to task vectors
        weighted_task_vectors = task_vectors_stacked * weights * elect_mask

        # Sum the weighted task vectors to get the final update
        merged_update = weighted_task_vectors.sum(dim=0)

        # Normalize by the sum of weights where the election mask is True
        normalization_factor = (weights * elect_mask.to(weights.dtype)).sum(dim=0).clamp(min=1e-10)  # Avoid division by zero
        normalized_merged_update = merged_update / normalization_factor

        # Compute the final merged model parameters
        merged_model_parameters = base_model_parameters + normalized_merged_update

        return merged_model_parameters



# EXMPLE CODE FOR TESTING

if __name__ == "__main__":
    # Example usage
    base_params = torch.tensor([0.0, 0.0, 0.0, 0.0])
    ft_params_1 = torch.tensor([1.0, -1.0, 0.5, -0.5])
    ft_params_2 = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    ft_params_3 = torch.tensor([0.5, 0.5, -1.0, -1.0])

    weights = [1.0, 1.0, 1.0]
    densities = [1.0, 1.0, 1.0]

    ties = TIES()
    merged_params = ties.merge(
        weights=weights,
        base_model_parameters=base_params,
        ft_models_parameters=[ft_params_1, ft_params_2, ft_params_3],
        densities=densities,
        device=torch.device("cpu"),
    )

    print("Merged Parameters:", merged_params)