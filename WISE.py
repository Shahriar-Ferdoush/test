import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2FN

from merging.merge import Merge

edit_history = []
merged_group_edit_history = []


def brackets_to_dots(s):
    s = s.replace("[", ".").replace("]", "")
    return s


def parent_module(model, module_name):
    """
    Given a model and a module name (e.g., "layer1.0.conv1"), return the parent module.
    """
    components = module_name.split(".")
    parent = model

    for comp in components[:-1]:
        if hasattr(parent, comp):
            parent = getattr(parent, comp)
        else:
            parent = parent[int(comp)]

    return parent


def euclidean_distance(tensor1, tensor2, config, activation_mask=None, infer=False):
    # Get the config activation function
    act_fn = ACT2FN[config.hidden_act]

    l2_norm = torch.norm(act_fn(tensor1) - act_fn(tensor2), dim=-1)

    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if activation_mask is not None:
        return torch.sum(l2_norm * activation_mask, dim=-1) / torch.sum(
            activation_mask, dim=-1
        )
    else:
        return torch.mean(l2_norm, dim=-1)


class EditingMeanActivation:
    def __init__(self, min_activation=1e9):
        self.reset(min_activation=min_activation)

    def reset(self, min_activation=1e9):
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.min_activation = min_activation

    def update(self, value):
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count
        self.min_activation = min(self.min_activation, value)

    def get_mean_activation(self):
        return self.avg

    def get_min_activation(self):
        return self.min_activation


class WISEMemoryLayer(torch.nn.Module):
    def __init__(self, config, layer, device):
        super(WISEMemoryLayer, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device if device is None else device
        self.config = config

        self.new_weight = torch.nn.Parameter(copy.deepcopy(self.weight))
        self.original_layer = copy.deepcopy(self.layer)

        # These two are not needed so far
        self.memory_wight = []
        self.memory_mean_activation = []

        assert not self.weight.requires_grad, print(
            "Error: The original layer weights should not require gradients."
        )

        self.merge_count = 0  # only for retrieve

        self.user_mask = None
        self.training = False
        self.editing = False

        self.editing_mean_activation = EditingMeanActivation()
        self.editing_total_count = 0

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weights(self):
        self.memory_wight.append(copy.deepcopy(self.new_weight))
        self.new_weight = copy.deepcopy(self.original_layer.weight)

        if self.config.retrieve:
            self.memory_mean_activation.append(
                copy.deepcopy(self.editing_mean_activation.mean_activation)
            )
            self.editing_mean_activation = EditingMeanActivation()

    def merge_weight(self):
        merger = Merge(merger=self.config.merge_alg)

        # if self.original_layer.weight.equals(self.layer.weight):
        if torch.equal(self.original_layer.weight, self.layer.weight):
            current_new_weight = merger.merge(
                weights=[self.config.weights / len(self.memory_wight)]
                * len(self.memory_wight),
                base_model_parameters=self.original_layer.weight,
                ft_models_parameters=self.memory_wight,
                densities=[self.config.densities] * len(self.memory_wight),
                device=self.memory_wight[0].device,
            )

        else:
            current_new_weight = merger.merge(
                weights=[(0.4 * self.config.weights) / len(self.memory_wight)]
                * len(self.memory_wight)
                + [0.6],
                base_model_parameters=self.original_layer.weight,
                ft_models_parameters=self.memory_wight + [self.layer.weight],
                densities=[self.config.densities] * len(self.memory_wight) + [1.0],
                device=self.memory_wight[0].device,
            )

        self.layer.weight = torch.nn.Parameter(
            current_new_weight.to(self.layer.weight.device),
            requires_grad = False,
        )
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        del self.memory_wight
        self.memory_wight = []

    def save_editing_activation(self):
        in_scope_distance = euclidean_distance(
            self.original_layer_output[:-1, ...],
            self.edited_layer_output[:-1, ...],
            self.config,
        )
        self.editing_mean_activation.update(in_scope_distance.mean().item())

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice(
            [0, 1],
            size=p_grad.size()[0],
            p=[mask_ratio, 1 - mask_ratio],
        )
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def new_weight_forward(self, input: Tensor, weight) -> Tensor:
        return F.linear(input, weight)

    def mask_edited_weight_gradient(self):
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.device)

    def forward(self, *args, **kwargs):
        if self.editing:
            layer_out = self.new_weight_forward(*args, weight=self.new_weight)
            self.original_layer_output = self.original_layer(*args)
            self.edited_layer_output = layer_out
        else:
            original_layer_output = self.original_layer(*args)
            layer_output = self.layer(*args)
            edited_layer_output = self.new_weight_forward(*args, weight=self.new_weight)

            delta_act_merge = euclidean_distance(
                original_layer_output,
                layer_output,
                self.config,
                infer=True,
            )

            delta_act_new_weight = euclidean_distance(
                original_layer_output,
                edited_layer_output,
                self.config,
                infer=True,
            )

            threshold_epsilon = (
                self.editing_mean_activation.get_min_activation()
                * self.config.activation_ratio
            )

            if (
                delta_act_merge.item() < threshold_epsilon
                and delta_act_new_weight.item() < threshold_epsilon
            ):
                layer_out = original_layer_output
            elif delta_act_merge.item() >= delta_act_new_weight.item():
                layer_out = layer_output
            else:
                layer_out = edited_layer_output

        return layer_out


class WISE(torch.nn.Module):
    def __init__(self, config, model, device):
        print("===================== Initializing WISE =================")
        super(WISE, self).__init__()

        self.config = config
        self.model = model
        self.device = device
        main_memory_layer_name_raw = config.inner_params[0]

        self.config.hidden_act = (
            self.model.config.hidden_act
            if hasattr(self.model.config, "hidden_act")
            else self.model.config.activation_function
        )
        print("Using activation function:", self.config.hidden_act)

        self.side_memory = None
        self.main_memory = None

        # Lets say we want to add perallel memory to "model.layers[27].mlp.down_proj.weight"
        # main_memory_layer_name_raw = "model.layers.27.mlp.down_proj.weight"
        # self.main_memory_layer = "model.layers.27.mlp.down_proj"
        seffixes = [".weight", ".bias"]
        self.main_memory_layer = (
            main_memory_layer_name_raw.rsplit(".", 1)[0]
            if any(main_memory_layer_name_raw.endswith(s) for s in seffixes)
            else main_memory_layer_name_raw
        )

        # Freeze the model parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.edit_module = parent_module(
            self.model, brackets_to_dots(self.main_memory_layer)
        )
        print("Editing layer:", self.main_memory_layer)
        print("Editing module:", self.edit_module)
        print("Editing module type:", type(self.edit_module))
        self.target_layer_name = self.main_memory_layer.rsplit(".", 1)[-1]

        side_memory_layer = getattr(self.edit_module, self.target_layer_name)
        # Print side memory and its type
        print("Target layer: ", self.target_layer_name, type(side_memory_layer))

        if type(side_memory_layer) is not WISEMemoryLayer:
            setattr(
                self.edit_module,
                self.target_layer_name,
                WISEMemoryLayer(
                    config,
                    side_memory_layer,
                    device,
                ),
            )
            self.original_layer = copy.deepcopy(side_memory_layer)

        torch.cuda.empty_cache()

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def get_side_memory_layer(self):
        side_memory_layer = getattr(self.edit_module, self.target_layer_name)

        assert type(side_memory_layer) is WISEMemoryLayer, print(
            f"Error: The layer {self.main_memory_layer} is not a WISESideMemories layer. Current type: {type(side_memory_layer)}"
        )
        return side_memory_layer

    def __calculate_ft_loss(self, tokens, last_prompt_token_loc):
        print("======================= Calculating FT Loss ==================")
        print("Model backprop mode:", self.model.training)

        # Forward pass in the model to get the logits
        logits = self.model(**tokens).logits
        print("Logits shape:", logits.shape)

        # Last prompt of the batch is localization prompt, thus, no role in loss
        loc_prompt_count = 1

        batch_size = tokens["input_ids"].shape[0] - loc_prompt_count
        # Actual batch size excluding last k sequences

        # We are using tokens["input_ids"] as model input
        # Tokens["labels"] as target output
        # They are both from the same sentence and sequence length and of same shape
        # Fist token of the sequence will therefore -> In correct case output the second token of the labels
        # Last token of the ["input_ids"] will not have a label corresponding to it
        # So from logits we remove the last token prediction
        _logits = logits[:-loc_prompt_count, :-1, :].contiguous()
        # From the labels we remove the first token as it is the random start token of the LLM
        _labels = tokens["labels"][:-loc_prompt_count, 1:].contiguous()
        # (B-k, S-1) = labels excluding last k sequences and first token

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        loss = loss_fct(
            _logits.view(-1, _logits.size(-1)),
            _labels.view(-1),
        )
        loss = loss.view(batch_size, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-loc_prompt_count]):
            label_mask[i, col_index - 1 :] = True

        ft_loss = ((loss * label_mask).sum(dim=1) / label_mask.sum(dim=1)).mean()
        print("FT Loss:", ft_loss.item())

        return ft_loss



    def __calculate_activation_loss(
        self,
        original_layer_output,
        edited_layer_output,
        config=None,
        activation_mask=None,
        deactivation_mask=None,
    ):
        loc_prompt_count = 1
        print("=================== Calculating Activation Loss =================")
        
        # Print some part of the activation masks
        print("Activation mask sample:", activation_mask.flatten()[:10] if activation_mask is not None else "None")
        print("Deactivation mask sample:", deactivation_mask.flatten()[:10] if deactivation_mask is not None else "None")

        if activation_mask is not None:
            in_scope_distance = euclidean_distance(
                original_layer_output[:-loc_prompt_count, ...],
                edited_layer_output[:-loc_prompt_count, ...],
                config,
                activation_mask=activation_mask,
            )

            out_scope_distance = euclidean_distance(
                original_layer_output[:-loc_prompt_count, ...],
                edited_layer_output[:-loc_prompt_count, ...],
                config,
                activation_mask=deactivation_mask,
            )

        else:
            in_scope_distance = euclidean_distance(
                original_layer_output[:-loc_prompt_count, ...],
                edited_layer_output[:-loc_prompt_count, ...],
                config,
            )

            out_scope_distance = euclidean_distance(
                original_layer_output[:-loc_prompt_count, ...],
                edited_layer_output[:-loc_prompt_count, ...],
                config,
            )
        print("In-scope distance:", in_scope_distance.mean().item())
        print("Out-scope distance:", out_scope_distance.mean().item())

        loss_margin = abs(out_scope_distance - in_scope_distance) + config.gamma
        loss_upper = out_scope_distance - config.alpha
        loss_lower = config.beta - in_scope_distance

        loss_margin = (
            torch.mean(loss_margin[loss_margin > 0])
            if min(loss_margin[loss_margin > 0].shape) > 0
            else torch.tensor(0.0).to(original_layer_output.device)
        )
        loss_upper = (
            torch.mean(loss_upper[loss_upper > 0])
            if min(loss_upper[loss_upper > 0].shape) > 0
            else torch.tensor(0.0).to(original_layer_output.device)
        )
        loss_lower = (
            torch.mean(loss_lower[loss_lower > 0])
            if min(loss_lower[loss_lower > 0].shape) > 0
            else torch.tensor(0.0).to(original_layer_output.device)
        )

        return loss_margin + loss_upper + loss_lower

    def edit(
        self,
        config,
        tokens,
        activation_mask=None,
        deactivation_mask=None,
    ):
        print("====================== Starting WISE Editing =================")
        global edit_history
        global merged_group_edit_history

        edit_history.append(
            [{f"{key}": value.to("cpu") for key, value in tokens.items()}, False]
        )

        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        # SETUP MEMORY and TRAINING FLAGS
        setattr(eval(f"self.model.{self.main_memory_layer}"), "training", True)
        print("Training mode:", getattr(eval(f"self.model.{self.main_memory_layer}"), "training"))

        setattr(eval(f"self.model.{self.main_memory_layer}"), "editing", True)
        print("Editing mode:", getattr(eval(f"self.model.{self.main_memory_layer}"), "editing"))

        self.get_side_memory_layer().set_parameter_tunable()
        print("Parameter tunable:", self.get_side_memory_layer().new_weight.requires_grad)

        if (
            getattr(eval(f"self.model.{self.main_memory_layer}"), "editing_total_count")
            % self.config.save_freq
            == 0
        ):
            print("Creating another mask per save frequency")
            # TODO: There is a issue here, there should be two value,
            self.get_side_memory_layer().generate_activation_mask(
                self.config.mask_ratio
            )

        print("=================== WISE Training Loop ==================")
        # Training loop
        for i in range(config.num_train_steps):
            print(f"--- Iteration {i+1}/{config.num_train_steps} ---")
            if i == 0:
                # Create the optimizer
                optimizer = torch.optim.SGD(
                    [self.get_side_memory_layer().new_weight],
                    lr=config.edit_lr,
                    weight_decay=1e-5,
                )

            ft_loss = self.__calculate_ft_loss(
                tokens,
                last_prompt_token_loc,
            )
            print("FT Loss:", ft_loss.item())

            act_loss = self.__calculate_activation_loss(
                self.get_side_memory_layer().original_layer_output,
                self.get_side_memory_layer().edited_layer_output,
                config=config,
                activation_mask=activation_mask,
                deactivation_mask=deactivation_mask,
            )
            print("Activation Loss:", act_loss.item())

            loss = ft_loss + act_loss.to(ft_loss.device)

            if i == config.n_iter - 1:
                self.get_side_memory_layer().save_editing_activation()

            optimizer.zero_grad()
            # print if loss backward is possible
            print("Loss requires grad:", loss.requires_grad)
            loss.backward()

            self.get_side_memory_layer().mask_edited_weight_gradient()
            optimizer.step()

        editing_total_count = (
            getattr(
                eval(f"self.model.{self.main_memory_layer}"), "editing_total_count"
            )
            + 1
        )
        setattr(
            eval(f"self.model.{self.main_memory_layer}"),
            "editing_total_count",
            editing_total_count,
        )

        if (
            self.config.save_freq is not None
            and editing_total_count % self.config.save_freq == 0
        ):
            self.get_side_memory_layer().save_weights()

        if editing_total_count % self.config.merge_freq == 0:
            self.get_side_memory_layer().merge_weight()

    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.main_memory_layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)