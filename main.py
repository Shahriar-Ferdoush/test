import random
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from WISE import WISE
from WISEconfig import WISEConfig

CONTEXT_TEMPLATES_CACHE = None


LOC_PROMPTS = [
    "nq question: What is the capital of France?",
    "nq question: Who is the president of the United States?",
    "nq question: What is the largest mammal?",
    "nq question: How many continents are there on Earth?",
    "nq question: What is the boiling point of water?",
    "nq question: Who wrote 'Romeo and Juliet'?",
]


def find_sublist_start_index(list_1, list_2):
    for i in range(len(list_1) - len(list_2) + 1):
        if all(x == y for x, y in zip(list_1[i : i + len(list_2)], list_2)):
            return i
    return None


def get_augmentation_templates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lenght_params: List[List[int]],
    device: str,
) -> List[str]:
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = []
        prompt_tokens = tokenizer(
            ["I", "You", "Because", "Yes", "Q: "],
            padding=True,
            return_tensors="pt",
        ).to(device)

        for length, n_gen in lenght_params:
            gen_token = model.generate(
                input_ids=prompt_tokens["input_ids"],
                attention_mask=prompt_tokens["attention_mask"],
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tokenizer.eos_token_id,
            )
            CONTEXT_TEMPLATES_CACHE += tokenizer.batch_decode(
                gen_token, skip_special_tokens=True
            )
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            context.replace("{", "{{").replace("}", "}}") + " {}"
            for context in CONTEXT_TEMPLATES_CACHE
        ]

    return CONTEXT_TEMPLATES_CACHE


def tokenize(
    updates,
    tokenizer: AutoTokenizer,
    augmentation_templates: List[str],
    config: WISEConfig,
    device: str,
) -> Tuple[Any, Any, Any]:
    tokens = []

    # Extract prompts and targets from updates
    prompt, label = updates["prompt"], updates["target"]
    if not isinstance(prompt, list):
        prompt = [prompt]
    if not isinstance(label, list):
        label = [label]

    updates["localization_prompt"] = random.choice(LOC_PROMPTS)
    mask_token = -100

    only_prompts = [
        f"{template.format(p)}" for p in prompt for template in augmentation_templates
    ]

    full_prompts = [
        f"{template.format(p + ' ' + l)}"
        for p, l in zip(prompt, label)
        for template in augmentation_templates
    ]

    full_prompts += [updates["localization_prompt"]]

    only_prompt_tokens = tokenizer(
        only_prompts,
        padding=True,
        return_tensors="pt",
        truncation=True,
    )

    full_prompt_tokens = tokenizer(
        full_prompts,
        padding=True,
        return_tensors="pt",
        truncation=True,
    )

    only_prompt_ids = only_prompt_tokens["input_ids"]
    num_only_prompt_tokens = [len(p) for p in only_prompt_ids]

    full_prompt_tokens["labels"] = full_prompt_tokens["input_ids"].clone()

    # Mask the labels for non-edit parts
    if getattr(config, "objective_optimization", "only_label") == "only_label":
        for i, length in enumerate(num_only_prompt_tokens):
            full_prompt_tokens["labels"][i][:length] = mask_token

    # Mask the padding tokens in the labels
    full_prompt_tokens["labels"][
        full_prompt_tokens["input_ids"] == tokenizer.pad_token_id
    ] = mask_token

    if updates["localization_prompt"] in updates["prompt"]:
        subject_token_with_space = tokenizer.encode(
            " " + updates["localization_prompt"], add_special_tokens=False
        )
        subject_token = tokenizer.encode(
            updates["localization_prompt"], add_special_tokens=False
        )
        subjext_length = len(subject_token_with_space)

        activation_mask = torch.zeros_like(full_prompt_tokens["input_ids"][:-1])
        deactivation_mask = torch.ones_like(full_prompt_tokens["input_ids"][:-1])

        for i, token in enumerate(full_prompt_tokens["input_ids"]):
            start_index = find_sublist_start_index(
                full_prompt_tokens.detach().cpu().numpy.tolist(),
                subject_token_with_space,
            )
            if start_index is None:
                start_index = find_sublist_start_index(
                    full_prompt_tokens.detach().cpu().numpy().tolist(), subject_token
                )
                subject_length = len(subject_token)
            activation_mask[i][start_index : start_index + subjext_length] = 1
            deactivation_mask[i][start_index : start_index + subjext_length] = 0

    else:
        activation_mask = None
        deactivation_mask = None

    full_prompt_tokens = {
        f"{key}": value.to(device) for key, value in full_prompt_tokens.items()
    }

    return full_prompt_tokens, activation_mask, deactivation_mask


def edit_model_with_WISE(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    updates: List[Dict],
    config: WISEConfig,
    num_steps: int,
    edit_lr: float,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    # Update config with method inputs
    config.n_iter = num_steps
    config.edit_lr = edit_lr

    augmentation_templates = get_augmentation_templates(
        model, tokenizer, lenght_params=[[5, 5], [10, 5]], device=model.device
    )
    wise = WISE(model=model, config=config, device=model.device)

    tokens, act_mask, deact_mask = tokenize(
        updates,
        tokenizer=tokenizer,
        augmentation_templates=augmentation_templates,
        config=config,
        device=model.device,
    )

    wise.edit(
        config=config,
        tokens=tokens,
        activation_mask=act_mask,
        deactivation_mask=deact_mask,
    )

    wise.to("cpu")
    return wise, tokenizer
