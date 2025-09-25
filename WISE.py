import copy
import gc
import random

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.activations import ACT2FN

from merging.merge import Merge
from .utils import EarlyStopMeter, EditingMeanAct, brackets_to_periods, parent_module

# This file implements the Weight-based Isolated Subspace Editing (WISE) algorithm
# as described in the paper "Editing Large Language Models: Problems, Methods, and Opportunities"
#
# WISE operates by creating isolated subspaces within the weight matrices of FFN layers,
# allowing for targeted parameter updates while minimizing catastrophic forgetting.
#
# Key mathematical components:
# - Main memory (W_v): Original weight matrix
# - Side memory (W_v'): Edited weight matrix
# - Subspace isolation: Using binary masks M_i ∈ {0,1}^d where d is parameter dimension
# - Activation threshold (ε): min_{x_t ∈ D_edit} Δact(x_t) * act_ratio
#
# Algorithms 1 & 2 are implemented here: Editing Stage and Inference Stage

# Dictionary of different merging strategies for side memory weights
# Used in Algorithm 1 (step 7): "Use Ties-Merge in Equation 8 to update the final side memory"
#
# Mathematical formulation for Ties-Merge (GTA with magnitude and sum):
# W_v'_merged = W_v + Σ(w_i * (W_v'_i - W_v) * S_i) where:
# - W_v is the original weight matrix (main memory)
# - W_v'_i are the edited weight matrices (side memories)
# - w_i are the weights for combining different side memories
# - S_i are the sparsity masks for each side memory (derived from magnitude)

# Global storage for edit history used for replay mechanism
# Related to Algorithm 1's WISE-Retrieve functionality
#
# For WISE-Retrieve, we maintain:
# - edit_history: List of all individual edits [({tokens}, used_flag)]
# - merge_group_edit_history: List of groups of edits after merging
# These histories enable replay-based regularization to maintain performance on previously edited examples
edit_history = []
merge_group_edit_history = []


def euc(query, key, config, act_mask=None, infer=False):
    """
    Euclidean distance computation for measuring activation difference between
    original (main memory) and edited (side memory) activations.

    Mathematical formulation:
    Δact(x) = ||A(x) · (W_v' - W_v)||₂

    Where:
    - A(x): Activation function applied to input x
    - W_v: Original weights (main memory)
    - W_v': Edited weights (side memory)
    - ||·||₂: L2 norm operation

    This function implements the activation calculation used in:
    1. Algorithm 2 steps 3-4: For determining which memory to use during inference
    2. Algorithm 1 step 4: For updating the activation threshold ε

    Parameters:
    - query: Original activations from W_v (main memory)
    - key: New activations from W_v' (side memory)
    - act_mask: Optional mask to compute masked distance for specific positions
    - infer: Flag for inference mode optimization

    Returns:
    - Euclidean distance between activations (scalar or per-example)
    """
    # Apply activation function from model config
    act_fn = ACT2FN[config.hidden_act]

    # Compute L2 norm of difference between activated values
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)

    # Optimization for inference with large batches - take top-k only
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    # Apply masking if provided (for targeted distance computation)
    if act_mask is not None:
        return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)


class WISE(torch.nn.Module):
    """
    Main WISE implementation class that handles the editing and inference logic.

    WISE (Weight-based Isolated Subspace Editing) operates by:
    1. Creating k isolated subspaces within FFN weight matrices using random binary masks
    2. Editing each subspace with a specific (x_t, y_t) example while minimizing interference
    3. Merging subspaces when filled using mathematical operations like Ties-Merge
    4. During inference, selecting between main memory (W_v) and side memory (W_v')
       based on activation differences

    This class implements both:
    - Algorithm 1 (Editing Stage): Modifying weights to incorporate new knowledge
    - Algorithm 2 (Inference Stage): Selectively using memories based on input

    WISE variants:
    - Standard WISE: Single side memory with multiple subspaces
    - WISE-Retrieve: Multiple side memories with activation-based retrieval
    """

    def __init__(self, config, model, device):
        """
        Initialize the WISE editor with the model to edit and configuration

        Parameters:
        - config: Configuration object containing:
            - inner_params: Target layers to edit (FFN layers)
            - mask_ratio (ρ): Proportion of weights in each subspace
            - retrieve: Whether to use WISE-Retrieve variant
            - merge_alg: Algorithm for merging subspaces
            - act_ratio: Scaling factor for activation threshold
            - save_freq: Frequency of subspace saving
            - merge_freq: Frequency of subspace merging
        - model: The language model to be edited
        - device: Computing device
        """

        """
        Trace:
        inner_params: ['transformer.h[8].mlp.c_fc.weight']
        model: "./hugging_cache/gpt2"
        
        """

        super(WISE, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, "hidden_act"):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, "activation_function"):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer
        # Getting the layer to edit
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        self.layer = (
            layer.rsplit(".", 1)[0]
            if any(layer.endswith(x) for x in suffixes)
            else layer
        )

        # Freeze all model parameters to prevent unintended updates
        for n, p in self.model.named_parameters():
            p.requires_grad = False

        # GPT2 uses Conv1D layers for projections instead of Linear layers
        if isinstance(
            self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
        ):
            conv1D = True
        else:
            conv1D = False

        # --- Add WISE to chosen layers ---
        # This creates the adapter that will contain both main memory W_v and side memory W_v'
        # The adapter intercepts forward passes to handle both editing and inference logic
        # For "model.layers[27].mlp.down_proj.weight"
        # edit module = model.layers[27].mlp.down_proj
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        # layer_name = "down_proj"

        adapter_layer = getattr(self.edit_module, self.layer_name)
        # adapter_layer = model.layers[27].mlp.down_proj

        if type(adapter_layer) is not WISEAdapter:
            # setattr(Object whose attribute is to be set, 'attribute name', value to be set)
            # setattr(model.layers[27].mlp, 'down_proj', WISEAdapter(config, adapter_layer, transpose=transpose))
            setattr(
                self.edit_module,
                self.layer_name,
                WISEAdapter(config, adapter_layer, conv1D=conv1D),
            )
            # edit_module = model.layers[27].mlp
            # model.layers[27].mlp.down_proj = WISEAdapter(config, model.layers[27].mlp.down_proj, transpose=transpose)
            self.original_layer = copy.deepcopy(adapter_layer)
            # original_layer = model.layers[27].mlp.down_proj = Wv
            print(f"New weights successfully inserted into {layer}")

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, **kwargs):
        """
        Model forward pass with WISE memory management

        This method relates to the end of Algorithm 1 (step 16): returning the final LLM model f_θ^T

        For standard WISE (not retrieve), this method performs final memory merges if needed:
        1. Checks if there are pending edits in the side memory
        2. If side memory differs from original and enough edits have accumulated (≥ save_freq),
           add the current side memory to memory_weight
        3. If memory_weight contains weights to merge, calls merge_weight() to consolidate them

        The merged weights implement Equation 8 from the paper:
        W_v'_merged = W_v + Σ(w_i * (W_v'_i - W_v) * S_i)

        Returns:
            Model outputs with properly selected weights (main or side memory)
        """
        if not self.config.retrieve:
            # If not in retrieve mode, perform final merges as needed
            if (
                hasattr(self.get_adapter_layer(), "editing")  # See if editing attribute exists
                and not self.get_adapter_layer().editing # Not currently editing
            ):
                # final merge
                if (
                    not self.get_adapter_layer().original_layer.weight.equal(
                        self.get_adapter_layer().new_weight
                    )
                    and self.get_adapter_layer().editing_total_cnt
                    >= self.config.save_freq
                ):
                    self.get_adapter_layer().memory_weight.append(
                        self.get_adapter_layer().new_weight
                    )
                if (
                    len(self.get_adapter_layer().memory_weight) > 0
                    and self.get_adapter_layer().editing_total_cnt
                    >= self.config.save_freq
                ):
                    print(
                        "length of memory is ",
                        len(self.get_adapter_layer().memory_weight),
                        "!!!!!!",
                    )
                    self.get_adapter_layer().merge_weight()
        return self.model(**kwargs)

    def reset_layer(self):
        """
        Reset the edited layer back to its original state

        This reverts any edits by replacing the WISEAdapter with the original layer,
        effectively setting W_v' = W_v (side memory = main memory)
        """
        layer = getattr(self.edit_module, self.layer_name)
        # layer = model.layers[27].mlp.down_proj
        del layer
        # delete the adapter layer to free memory
        setattr(
            self.edit_module, self.layer_name, self.get_adapter_layer().original_layer
            # setattr(model.layers[27].mlp, 'down_proj', original_layer)
            # model.layers[27].mlp.down_proj = original_layer
            # Edit module with base model parameters
        )

    def get_adapter_layer(self):
        """
        Helper to access the WISEAdapter that contains both main and side memories

        Returns:
            WISEAdapter: The adapter module that handles weight isolation and selection

        Raises:
            AssertionError: If adapter layer is not correctly added
        """
        adapter_layer = getattr(self.edit_module, self.layer_name)
        # adapter_layer = model.layers[27].mlp.down_proj
        assert type(adapter_layer) is WISEAdapter, print(
            "Adapter Layer is not added correctly...."
        )
        return adapter_layer

    # TODO: generation
    def generate(self, *args, **kwargs):
        """
        Model text generation with WISE-edited weights

        During generation, the proper memory (main or side) will be selected
        based on the activation difference (Δact) and threshold (ε) as in
        Algorithm 2 (Inference Stage)

        Returns:
            Generated text using dynamically selected weights
        """
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        """
        Core editing method implementing Algorithm 1: WISE Editing Stage

        Each call handles one edit sample (x_t, y_t) from D_edit:
        1. Applies subspace masks (M_i) to constrain weight updates
        2. Optimizes side memory (W_v') with the edit loss function
        3. Updates activation threshold (ε) based on Δact(x_t)
        4. Manages memory filling and merging when subspaces are full

        Mathematical components:
        - Edit loss L_edit = -log P_{W_v'}(y_t|x_t) + L_a
        - Activation loss L_a with margin objectives
        - Subspace isolation via binary masks M_i
        - Activation threshold ε = min(ε, Δact(x_t))

        Parameters:
            config: Configuration with optimization parameters
            tokens: Input-output token pairs (x_t, y_t)
            act_mask: Optional mask for targeted activation computation
            deact_mask: Optional mask for contrasting activation regions
        """
        # for retrieve ##
        # This part implements tracking for Algorithm 1, step 8-9 (WISE-Retrieve)
        global edit_history
        global merge_group_edit_history
        edit_history.append(
            [{f"{k1}": v1.to("cpu") for k1, v1 in tokens.items()}, False]
        )
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()

        # Algorithm 1, step 1: Generate random masks M_i of ratio ρ
        # The masks are regenerated periodically based on save_freq
        # Mathematically: M_i ∈ {0,1}^d where d is parameter dimension
        # and |M_i|_0/d = ρ (mask_ratio)
        if (
            getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt")
            % self.config.save_freq
            == 0
        ):
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

        # --- train Wise value ---
        # Algorithm 1, step 3: Edit in the memory subspace with loss L_edit
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):

            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD(
                    [self.get_adapter_layer().new_weight],
                    config.edit_lr,
                    weight_decay=1e-5,
                )

            # Compute NLL loss term: -log P_{W_v'}(y_t|x_t)
            # This measures how well the side memory produces the target output
            ft_loss = self.__cal_ft_loss(tokens, last_prompt_token_loc)

            # Compute activation loss term: L_a
            # This encourages the activations to change for edited inputs and
            # remain similar for unedited inputs (promoting specificity)
            act_loss = self.__cal_activation_loss(
                self.get_adapter_layer().original_layer_output,
                self.get_adapter_layer().new_weight_layer_output,
                config=config,
                act_mask=act_mask,
                deact_mask=deact_mask,
            )

            # Combined loss: L_edit = -log P_{W_v'}(y_t|x_t) + L_a
            # The full editing objective from Algorithm 1, step 3
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                # Algorithm 1, step 4: Update activation threshold ε = min(ε, Δact(x_t))
                # This tracks the minimum activation difference to determine when to use side memory
                self.get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                # Algorithm 1, step 4: Update activation threshold ε = min(ε, Δact(x_t))
                self.get_adapter_layer().save_editing_activation()  # add last gradient

            # This implements replay mechanism for WISE-Retrieve (see Appendix B.3)
            # Not directly in Algorithm 1 pseudocode but enhances retrieve mode with
            # contrastive objectives to maintain memory distinctiveness
            if (
                self.config.retrieve
                and self.get_adapter_layer().merge_cnt > 0
                and self.config.replay
            ):
                # Negative memory loss: encourage current memory to differ from past memories
                # Mathematical objective: max Δact for memories that should be distinct
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_):  ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {
                        f"{k1}": v1.to(self.config.device)
                        for k1, v1 in memo_input.items()
                    }
                    self.model(**memo_input)

                    # Add loss term that pushes different memory regions apart
                    # L_neg = max(0, 5 - Δact): penalize if activation difference too small
                    memory_act_loss = self.__cal_memory_neg_activation_loss(
                        self.get_adapter_layer().original_layer_output,
                        self.get_adapter_layer().new_weight_layer_output,
                        config=config,
                        act_mask=act_mask,
                        deact_mask=deact_mask,
                    )
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss

                if len(edit_history) > 0:
                    # Positive memory loss: encourage similarity for related memories
                    # Mathematical objective: min Δact for the same edit memory
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {
                        f"{k1}": v1.to(self.config.device)
                        for k1, v1 in memo_input.items()
                    }
                    self.model(**memo_input)

                    # Add loss term that pulls similar memory regions together
                    # L_pos = max(0, Δact - 20): penalize if activation difference too large
                    pos_memo_loss = self.__cal_memory_pos_activation_loss(
                        self.get_adapter_layer().original_layer_output,
                        self.get_adapter_layer().new_weight_layer_output,
                        config=config,
                        act_mask=act_mask,
                        deact_mask=deact_mask,
                    )
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            # Apply mask to gradients to ensure editing happens only in the selected subspace
            # This enforces W_v'[i] = W_v[i] for all i where M[i] = 0 (mask is 0)
            self.get_adapter_layer().mask_new_weight_gradient()

            if (
                self.config.retrieve
                and self.get_adapter_layer().merge_cnt > 0
                and self.config.replay
            ):
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            # Optional norm constraint to keep edited weights close to original
            # This restricts the magnitude of weight changes: ||W_v' - W_v||_∞ ≤ norm_constraint
            if type(self.config.norm_constraint) is float:
                self.__norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = (
            getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        )
        setattr(
            eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt
        )

        # Algorithm 1, steps 5-11: Check subspace fullness and handle memory management
        # Save weights to memory when a subspace is full (based on save_freq)
        if (
            self.config.save_freq is not None
            and editing_total_cnt % self.config.save_freq == 0
        ):
            # If current subspace M_i is full, move to next subspace M_{i+1}
            # This is implemented by saving the current weights to memory
            # and resetting the side memory to start fresh with a new mask
            self.get_adapter_layer().save_weight()
            print(f"Add New Weight to Memory...")

        # Algorithm 1, step 7: Use Ties-Merge to update final side memory
        # Algorithm 1, steps 8-10: If WISE-Retrieve, move to another copy
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            # In WISE-Retrieve, we track edit groups for replay-based regularization
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            # Apply mathematical merging operation (e.g., Ties-Merge from Equation 8)
            # W_v'_merged = W_v + Σ(w_i * (W_v'_i - W_v) * S_i)
            self.get_adapter_layer().merge_weight()
            print(
                f"Merge Weight of (New, Original) Matrix... with ties"
            )

    def __norm_constraint(self, norm_constraint):
        """
        Optional constraint to keep edited weights close to original
        Not explicitly in Algorithm 1 pseudocode, but helps with stability

        This implements an L∞ norm constraint on weight changes:
        ||W_v' - W_v||_∞ ≤ norm_constraint

        For each element: W_v'[i,j] ∈ [W_v[i,j] - norm_constraint, W_v[i,j] + norm_constraint]
        """
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight,
                min=original_weight - norm_constraint,
                max=original_weight + norm_constraint,
            )

    def __cal_ft_loss(self, tokens, last_prompt_token_loc):
        """
        Calculate fine-tuning loss (negative log-likelihood): -log P_{W_v'}(y_t|x_t)

        This implements the first term in Algorithm 1, step 3:
        Edit with L_edit = -log P_{W_v'}(y_t|x_t) + L_a

        The loss measures how well the edited model (with side memory W_v')
        predicts the target outputs y_t given inputs x_t.

        Mathematical formulation:
        L_NLL = -∑_{i ∈ target tokens} log(P(y_i|x, y_{<i}))

        Parameters:
            tokens: Input-output token pairs (x_t, y_t)
            last_prompt_token_loc: Location of last input token (to mask loss computation)

        Returns:
            NLL loss value (scalar)
        """
        k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits # (batch, seq_len, vocab_size)
        shift_logits = logits[:-k, :-1, :].contiguous() # 
        shift_labels = tokens["labels"][:-k, 1:].contiguous()

        # Create mask to only compute loss on target tokens (not input tokens)
        label_mask = torch.zeros_like(shift_labels, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1 :] = True

        shift_labels[~label_mask] = -100

        # Compute negative log probabilities
        log_probs = -nn.functional.log_softmax(shift_logits, dim=-1)

        if shift_labels.dim() == log_probs.dim() - 1:
            shift_labels = shift_labels.unsqueeze(-1)

        padding_mask = shift_labels.eq(-100)

        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        shift_labels = torch.clamp(shift_labels, min=0)

        # Gather the log probs of the target tokens
        nll_loss = log_probs.gather(dim=-1, index=shift_labels)
        nll_loss.masked_fill_(padding_mask, 0.0)

        # Normalize by the number of active elements
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements

        return nll_loss
        # loss_fct = CrossEntropyLoss(reduction='none')
        # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # loss = loss.view(bs, -1)

        # label_mask = torch.zeros_like(loss, dtype=torch.bool)

        # for i, col_index in enumerate(last_prompt_token_loc[:-k]):
        #     label_mask[i, col_index - 1:] = True

        # ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        # return ft_loss

    def __cal_activation_loss(
        self,
        original_layer_output,
        new_weight_layer_output,
        config=None,
        act_mask=None,
        deact_mask=None,
    ):
        """
        Calculate activation loss (L_a) component with multiple margin-based objectives

        This is part of Algorithm 1, step 3: Edit with L_edit = -log P_{W_v'}(y_t|x_t) + L_a

        The activation loss serves multiple purposes:
        1. Encourages high activation difference (Δact) for edited examples
        2. Constrains activation difference for unedited examples
        3. Helps compute the activation threshold ε = min(ε, Δact(x_t))

        Mathematical formulation:
        L_a = L_margin + L_upper + L_lower where:
        - L_margin = max(0, Δact_out - Δact_in + γ): Encourages separation
        - L_upper = max(0, Δact_out - α): Upper bound constraint
        - L_lower = max(0, β - Δact_in): Lower bound constraint

        Parameters:
            original_layer_output: Activations from original weights (W_v)
            new_weight_layer_output: Activations from edited weights (W_v')
            config: Configuration with hyperparameters (α, β, γ)
            act_mask: Optional mask for in-scope activations
            deact_mask: Optional mask for out-of-scope activations

        Returns:
            Combined activation loss (scalar)
        """
        k = 1
        if act_mask is not None:
            # Compute activation differences with masks for targeted regions
            in_scope_dist = euc(
                original_layer_output[:-k, ...],
                new_weight_layer_output[:-k, ...],
                config,
                act_mask=act_mask,
            )
            out_scope_dist = euc(
                original_layer_output[:-k, ...],
                new_weight_layer_output[:-k, ...],
                config,
                act_mask=deact_mask,
            )
        else:
            # Compute activation differences based on batch segments
            # In-scope: first part of batch (edited examples)
            in_scope_dist = euc(
                original_layer_output[:-k, ...],
                new_weight_layer_output[:-k, ...],
                config,
            )
            # Out-of-scope: last part of batch (unedited examples)
            out_scope_dist = euc(
                original_layer_output[-k:, ...],
                new_weight_layer_output[-k:, ...],
                config,
            )

        # Margin loss: ensures out-of-scope activations are higher than in-scope by margin γ
        # L_margin = max(0, Δact_out - Δact_in + γ)
        loss = out_scope_dist.view(-1, 1) - in_scope_dist + config.gamma

        # Upper bound loss: ensures out-of-scope activations don't exceed α
        # L_upper = max(0, Δact_out - α)
        loss2 = out_scope_dist - config.alpha

        # Lower bound loss: ensures in-scope activations are at least β
        # L_lower = max(0, β - Δact_in)
        loss3 = config.beta - in_scope_dist

        # Apply ReLU (max(0,·)) and compute means for each loss component
        loss3 = (
            torch.mean(loss3[loss3 > 0])
            if min(loss3[loss3 > 0].size()) > 0
            else torch.tensor(0.0).to(original_layer_output.device)
        )
        loss2 = (
            torch.mean(loss2[loss2 > 0])
            if min(loss2[loss2 > 0].size()) > 0
            else torch.tensor(0.0).to(original_layer_output.device)
        )
        loss = (
            torch.mean(loss[loss > 0])
            if min(loss[loss > 0].size()) > 0
            else torch.tensor(0.0).to(original_layer_output.device)
        )
        # Combine all loss components
        return loss + loss2 + loss3

    def __cal_memory_pos_activation_loss(
        self,
        original_layer_output,
        new_weight_layer_output,
        config=None,
        act_mask=None,
        deact_mask=None,
    ):
        """
        Calculate positive memory activation loss for replay in WISE-Retrieve

        Used in WISE-Retrieve with replay mechanism (Appendix B.3)
        This encourages similar activation patterns for related examples.

        Mathematical formulation:
        L_pos = max(0, Δact - η_pos)

        Where:
        - Δact: Activation difference between original and edited outputs
        - η_pos: Upper threshold for activation difference (20 in implementation)

        The loss penalizes when activation differences become too large
        for samples that should have similar behavior.

        Parameters:
            original_layer_output: Activations from original weights (W_v)
            new_weight_layer_output: Activations from edited weights (W_v')
            config: Configuration parameters
            act_mask: Optional mask for targeted activation computation
            deact_mask: Optional mask for contrasting activation regions

        Returns:
            Positive memory activation loss (scalar)
        """
        k = 1
        # Calculate activation difference between original and new weights
        in_scope_dist = euc(
            original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config
        )
        # Upper bound loss: penalize if activation difference exceeds η_pos = 20
        # L_pos = max(0, η_pos - Δact)
        loss4 = 20 - in_scope_dist

        # Apply ReLU (max(0,·)) and compute mean
        return (
            torch.mean(loss4[loss4 > 0])
            if min(loss4[loss4 > 0].size()) > 0
            else torch.tensor(0.0)
        )

    def __cal_memory_neg_activation_loss(
        self,
        original_layer_output,
        new_weight_layer_output,
        config=None,
        act_mask=None,
        deact_mask=None,
    ):
        """
        Calculate negative memory activation loss for replay in WISE-Retrieve

        Used in WISE-Retrieve with replay mechanism (Appendix B.3)
        This encourages different activation patterns for unrelated examples.

        Mathematical formulation:
        L_neg = max(0, η_neg - Δact)

        Where:
        - Δact: Activation difference between original and edited outputs
        - η_neg: Lower threshold for activation difference (5 in implementation)

        The loss penalizes when activation differences become too small
        for samples that should have different behavior, preventing
        interference between unrelated edits.

        Parameters:
            original_layer_output: Activations from original weights (W_v)
            new_weight_layer_output: Activations from edited weights (W_v')
            config: Configuration parameters
            act_mask: Optional mask for targeted activation computation
            deact_mask: Optional mask for contrasting activation regions

        Returns:
            Negative memory activation loss (scalar)
        """
        k = 1
        # Calculate activation difference between original and new weights
        in_scope_dist = euc(
            original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config
        )
        # Lower bound loss: penalize if activation difference is below η_neg = 5
        # L_neg = max(0, Δact - η_neg)
        loss4 = in_scope_dist - 5

        # Apply ReLU (max(0,·)) and compute mean
        return (
            torch.mean(loss4[loss4 > 0])
            if min(loss4[loss4 > 0].size()) > 0
            else torch.tensor(0.0)
        )


class WISEAdapter(torch.nn.Module):
    """
    WISE adapter that replaces the target FFN layer in the model

    This adapter implements:
    1. Weight isolation via subspace masks during editing
    2. Memory management for both standard WISE and WISE-Retrieve variants
    3. Activation-based weight selection during inference

    The adapter contains:
    - Main memory W_v: Original FFN weights
    - Side memory W_v': Edited weights in isolated subspaces
    - Multiple side memories for WISE-Retrieve
    - Activation tracking for threshold-based memory selection

    This class handles both:
    - Editing operations (Algorithm 1): Creating isolated subspaces, updating weights
    - Inference operations (Algorithm 2): Selecting between memories based on activations
    """

    def __init__(self, config, layer, conv1D):
        """
        Initialize the WISEAdapter that replaces a target layer

        Parameters:
            config: Configuration with WISE parameters
            layer: The original layer to be replaced (typically FFN)
            conv1D: Flag for handling 1D convolution layers (GPT-2)
        """
        super(WISEAdapter, self).__init__()
        #layer = model.layers[27].mlp.down_proj 

        self.layer = layer
        self.weight = self.layer.weight
        # self.weight = model.layers[27].mlp.down_proj.weight = Wv
        self.device = layer.weight.device
        self.config = config
        # Initialize side memory (W_v')
        self.new_weight = copy.deepcopy(self.weight)
        # Keep original layer (main memory Wv)
        self.original_layer = copy.deepcopy(self.layer)
        # Storage for multiple side memories (for Algorithm 1 steps 6-11)
        self.memory_weight = []
        self.memory_mean_act = []
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print(
            "Original Layer can not be tunable...."
        )

        self.used_mask = None

        self.training = False
        self.editing = False
        self.conv1D = conv1D

        # Track activation thresholds (for Algorithm 1 step 4 and Algorithm 2 steps 5-10)
        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

    def set_parameter_tunable(self):
        """
        Allow side memory weights to be updated during editing

        This enables gradient-based optimization of W_v' during edit training
        """
        self.new_weight.requires_grad = True

    def save_weight(self):
        """
        Save current side memory to memory list and reset for next subspace

        Implements Algorithm 1 steps 6-11 (subspace management):
        - Store current W_v' in memory list
        - Reset W_v' for potential next edit
        - For WISE-Retrieve, also store activation statistics

        This is called when a subspace is filled or when a new edit is needed
        in a different subspace.
        """
        # Edited weights appened to the memory_weight list
        self.memory_weight.append(copy.deepcopy(self.new_weight))

        # Reset side memory to original weights for next edit
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        """
        Merge side memories using the specified algorithm

        Implements Algorithm 1 step 7: Use specified merge algorithm
        to update final side memory W_v'

        Available merge algorithms:
        - ties-merge: Weighted interpolation with decay
        - slerp: Spherical linear interpolation
        - linear: Simple linear interpolation
        - gta: GTA-based merging
        """
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                # Standard WISE merging with selected algorithm
                merger = Merge(merger="ties")
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merger.edit(
                        [
                            self.config.weights / len(self.memory_weight)
                            for _ in range(len(self.memory_weight))
                        ],
                        self.original_layer.weight,
                        self.memory_weight,
                        densities=self.config.densities,
                    )
                else:
                    cur_new_weight = merger.edit(
                        [
                            0.4 / len(self.memory_weight)
                            for _ in range(len(self.memory_weight))
                        ]
                        + [0.6],
                        self.original_layer.weight,
                        self.memory_weight + [self.layer.weight],
                        densities=self.config.densities,
                    )
                self.layer.weight = torch.nn.Parameter(
                    cur_new_weight.to(self.layer.weight.device), requires_grad=False
                )
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                # WISE-Retrieve merging (Algorithm 1 steps 8-10)
                # For WISE-Retrieve, we merge every merge_freq / save_freq memories
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                merger = Merge(merger="ties")

                # Merge the most recent merge_num memories
                new_merge_weight = merger.edit(
                    [self.config.weights / merge_num for _ in range(merge_num)],
                    self.original_layer.weight,
                    self.memory_weight[-merge_num:],
                    densities=self.config.densities,
                )

                # Track minimum activation threshold across merged memories
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())

                # Reset and store the merged memory
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            # Simple merging when save_freq is None
            merger = Merge(merger="ties")
            cur_new_weight = merger.edit(
                0.5,  # Equal weighting between original and edited weights
                self.layer.weight,
                [self.new_weight],
                densities=self.config.densities,
            )
            self.layer.weight = torch.nn.Parameter(
                cur_new_weight.to(self.layer.weight.device), requires_grad=False
            )
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def save_editing_activation(self):
        """
        Update activation threshold based on current edit

        Implements Algorithm 1 step 4: Update activation threshold ε = min(ε,Δact(xt))

        This captures the minimum activation difference between original and edited
        weights for the current edit, which becomes the threshold for future inference.
        """
        # Calculate activation difference (Δact) between original and edited weights
        in_scope_dist = euc(
            self.original_layer_output[:-1, ...],
            self.new_weight_layer_output[:-1, ...],
            self.config,
        )
        # Update activation threshold with mean activation difference
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_activation_mask(self, mask_ratio):
        """
        Generate random mask for subspace selection

        Implements Algorithm 1 step 1: Generate k random masks M_i of ratio ρ
        where each mask contains randomly selected elements to be modified.

        Parameters:
            mask_ratio: Ratio of elements to be selected (ρ in Algorithm 1)
        """
        p_grad = self.new_weight.reshape(-1)
        # Randomly select elements with probability mask_ratio
        p_mask = np.random.choice(
            [1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio]
        )
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        """
        Generate non-overlapping masks for subspace selection

        Alternative implementation for Algorithm 1 step 1 that ensures
        different edits use different subspaces by tracking used mask positions

        Parameters:
            mask_ratio: Ratio of elements to be selected (ρ in Algorithm 1)
        """
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])

        # Initialize tracking mask if not yet created
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)

        # Get indices of elements that haven't been used in previous masks
        available_indices = np.where(~self.used_mask)[0]

        # Check if we have enough unused elements
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")

        # Randomly select from available indices
        chosen_indices = np.random.choice(
            available_indices, size=mask_size, replace=False
        )

        # Create binary mask with 1s at chosen positions
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1

        # Mark selected indices as used for future masks
        self.used_mask[chosen_indices] = True

        # Convert to tensor and store
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def new_weight_forward(self, input: Tensor, weight) -> Tensor:
        """
        Compute forward pass with a given weight matrix

        Handles both convolution layers (GPT-2) and linear layers (other models)

        Parameters:
            input: Input tensor
            weight: Weight matrix to use for forward pass

        Returns:
            Output activation tensor
        """
        if self.conv1D:
            # Handle conv1D case (e.g., GPT-2)
            size_out = input.size()[:-1] + (weight.size(1),)
            input = torch.addmm(
                self.original_layer.bias, input.view(-1, input.size(-1)), weight
            )
            input = input.view(size_out)
            return input
        else:
            # Standard linear layer
            return F.linear(input, weight)

    def mask_new_weight_gradient(self):
        """
        Apply mask to gradients to ensure editing happens only in the selected subspace

        Implements part of Algorithm 1 step 3: Constraining gradient updates to occur
        only within the masked subspace M_i by zeroing out gradients outside the mask

        This ensures that editing is localized to the selected subspace, preserving
        model behavior on unrelated inputs.
        """
        assert self.new_weight.grad is not None, print(
            "Gradient Collection for New Weight error, gradient not found"
        )
        # Apply gradient mask to constrain updates to the selected subspace
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # Element-wise multiplication with mask (0s outside the subspace, 1s inside)
        p_grad = p_grad * self.weight_mask
        # Reshape back to original gradient shape
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def forward(self, *args):
        """
        Forward pass that handles both editing and inference

        During editing mode:
        - Computes both original (W_v) and new (W_v') memory outputs
        - Stores these for loss calculation between them

        During inference mode:
        - Implements Algorithm 2 (Inference Stage)
        - Computes activation differences to determine memory selection
        - Dynamically selects between memories based on activation threshold

        Parameters:
            *args: Input tensors passed to the layer

        Returns:
            Output tensor from selected weights
        """
        if self.editing:
            # During editing: compute outputs for both memories to calculate losses
            layer_out = self.new_weight_forward(*args, self.new_weight)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            if not self.config.retrieve:
                # Standard WISE inference (Algorithm 2)
                # Compute outputs from all weight matrices
                original_layer_output = self.original_layer(*args)
                layer_output = self.layer(*args)
                new_weight_layer_output = self.new_weight_forward(
                    *args, self.new_weight
                )

                # Algorithm 2 steps 3-4: Compute activation differences
                # Δact = ||A(x) · (W_v' - W_v)||₂
                dist2 = euc(
                    original_layer_output,
                    new_weight_layer_output,
                    self.config,
                    infer=True,
                )
                dist1 = euc(
                    original_layer_output, layer_output, self.config, infer=True
                )

                # Algorithm 2 steps 5-10: Threshold-based memory selection
                # Use threshold ε * act_ratio to decide which memory to use
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                # Apply memory selection logic (Algorithm 2 steps 7-10)
                if dist1.item() < threshold and dist2.item() < threshold:
                    # Case 1: All activation differences below threshold
                    # Use original memory (W_v) as default
                    layer_out = original_layer_output
                elif dist1.item() > dist2.item():
                    # Case 2: Layer memory has higher activation difference
                    # Use the intermediate merged memory
                    layer_out = layer_output
                else:
                    # Case 3: New weight memory has higher activation difference
                    # Use the side memory (W_v')
                    layer_out = new_weight_layer_output
            else:
                # WISE-Retrieve inference with multiple memory copies
                original_layer_output = self.original_layer(*args)
                new_weight_layer_output = self.new_weight_forward(
                    *args, self.new_weight
                )

                # Initialize with comparison between original and current edited memory
                dist1 = euc(
                    original_layer_output,
                    new_weight_layer_output,
                    self.config,
                    infer=True,
                )
                # Get activation threshold for this memory set
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio
                min_dist = dist1

                # First check if we should use original memory (below threshold)
                if min_dist.item() < threshold:
                    layer_out = original_layer_output
                else:
                    # Start with current edited memory as default
                    layer_out = new_weight_layer_output

                # Check all stored memory copies to find the one with maximal activation
                for i in range(len(self.memory_weight)):
                    # Get the i-th memory copy
                    memory_retrieve_weight = self.memory_weight[i]
                    # Compute its output
                    memory_weight_layer_output = self.new_weight_forward(
                        *args, memory_retrieve_weight
                    )
                    # Calculate activation difference
                    dist = euc(
                        original_layer_output,
                        memory_weight_layer_output,
                        self.config,
                        infer=True,
                    )
                    # Memory selection logic: if this memory has higher activation
                    # and exceeds its threshold, select it
                    if (
                        dist > min_dist
                        and dist
                        > self.memory_mean_act[i].min_act() * self.config.act_ratio
                    ):
                        layer_out = memory_weight_layer_output
                        min_dist = dist
                    print(
                        dist, self.memory_mean_act[i].min_act() * self.config.act_ratio
                    )
        return layer_out
