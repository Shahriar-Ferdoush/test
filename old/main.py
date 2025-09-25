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
            ["I", "nq question: ", "Because", "Yes", "Q: "],
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

    len_temp = len(augmentation_templates)
    prompts = [item["prompt"] for item in updates]
    labels = [item["target"] for item in updates]
    loc_prompts = (
        [item["loc_prompt"] for item in updates] if "loc_prompt" in updates[0] else []
    )

    mask_token = -100
    if hasattr(config, "use_chat_templates") and config.use_chat_templates:
        full_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": temp.format(p)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            + " "
            + l
            for temp in augmentation_templates
            for p, l in zip(prompts, labels)
        ]
        promtp_ids = tokenizer(
            [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": temp.format(p)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for temp in augmentation_templates
                for p in prompts
            ],
            padding=True,
            return_tensors="pt",
            truncation=True,
        )["input_ids"]

    else:
        full_prompts = [
            temp.format(p) + " " + l
            for temp in augmentation_templates
            for p, l in zip(prompts, labels)
        ]
        promtp_ids = tokenizer(
            [temp.format(p) for temp in augmentation_templates for p in prompts],
            padding=True,
            return_tensors="pt",
            truncation=True,
        )["input_ids"]
    full_prompts += loc_prompts

    num_prompt_toks = [len(i) for i in promtp_ids]
    tokens = tokenizer(
        full_prompts, padding=True, return_tensors="pt", truncation=True
    )
    tokens[labels] = tokens["input_ids"].clone()

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    activation_masks = []
    deactivation_masks = []

    for i, loc_prompt in enumerate(loc_prompts):
        if loc_prompt in prompts[i]:
            subject_token = tokenizer.encode(" "+ loc_prompt, add_special_tokens=False)
            subject_token1 = tokenizer.encode(loc_prompt, add_special_tokens=False)

            subject_lenght = len(subject_token)

            activation_mask = torch.zeros_like(
                tokens["input_ids"][int(i*len_temp): int((i+1)*len_temp)]
            )
            deactivation_mask = torch.zeros_like(
                tokens["input_ids"][int(i*len_temp): int((i+1)*len_temp)]
            )

            for j, token in enumerate(
                tokens["input_ids"][int(i*len_temp): int((i+1)*len_temp)]
            ):
                start_index = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token)
                if start_index is None:
                    start_index = find_sublist_start_index(
                        token.detach().cpu().numpy().tolist(), subject_token1
                    )
                    subject_lenght = len(subject_token1)
                activation_mask[j, start_index : start_index + subject_lenght] = 1
                deactivation_mask[j][: start_index] = 1
                deactivation_mask[j][start_index + subject_lenght :] = 1
        else:
            activation_mask = None
            deactivation_mask = None
        activation_masks.append(activation_mask)
        deactivation_masks.append(deactivation_mask)
    
    activation_masks = [mask.to(device) if mask is not None else None for mask in activation_masks]
    deactivation_masks = [mask.to(device) if mask is not None else None for mask in deactivation_masks]

    tokens = {key: val.to(device) for key, val in tokens.items()}

    return tokens, activation_mask, deactivation_mask


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

    print(f"Number of augmentation templates: {len(augmentation_templates)}")
    for temp in augmentation_templates:
        print(f"Template: {temp}")

    wise = WISE(model=model, config=config, device=model.device)

    tokens, act_mask, deact_mask = tokenize(
        updates,
        tokenizer=tokenizer,
        augmentation_templates=augmentation_templates,
        config=config,
        device=model.device,
    )
    print("Tokens prepared for editing.")
    print(f"Input IDs shape: {tokens['input_ids'].shape}")

    # Print all tokens and activation masks for debugging
    print("Input IDs:\n", tokens["input_ids"])
    print("Attention Mask:\n", tokens["attention_mask"])
    print("Labels:\n", tokens["labels"])

    print("Activation Mask:\n", act_mask if act_mask is not None else "None")
    print("Deactivation Mask:\n", deact_mask if deact_mask is not None else "None")

    wise.edit(
        config=config,
        tokens=tokens,
        activation_mask=act_mask,
        deactivation_mask=deact_mask,
    )

    wise.to("cpu")
    return wise, tokenizer
