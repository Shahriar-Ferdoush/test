import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT_TEMPLATES_CACHE = None


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
        decoded_tokens = tokenizer.batch_decode(gen_token, skip_special_tokens=True)

        # Escape any existing curly braces in the templates to avoid format errors
        escaped_tokens = [
            re.sub(r"{", "{{", re.sub(r"}", "}}", token)) for token in decoded_tokens
        ]
        CONTEXT_TEMPLATES_CACHE += escaped_tokens

    # Add the empty template and the templates with placeholder
    CONTEXT_TEMPLATES_CACHE = ["{}"] + [
        context + " {}" for context in CONTEXT_TEMPLATES_CACHE
    ]

    return CONTEXT_TEMPLATES_CACHE


def tokenize(batch, tokenizer, device, context_templates=None, hparams=None):
    # Initialize lists to store the processed data from each batch entry
    len_temp = len(context_templates)
    prompts = [item["prompt"] for item in batch]
    labels = [item["target_new"] for item in batch]
    loc_prompts = [item["loc_prompt"] for item in batch]

    mask_token = -100  # ignore_index of CrossEntropyLoss
    if hasattr(hparams, "use_chat_template") and hparams.use_chat_template:
        full_prompt = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": templ.format(p)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            + " "
            + l
            for templ in context_templates
            for p, l in zip(prompts, labels)
        ]
        prompt_ids = tokenizer(
            [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": templ.format(p)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for templ in context_templates
                for p in prompts
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"]
    else:
        full_prompt = [
            f"{templ.format(p + ' ' + l)}"
            for templ in context_templates
            for p, l in zip(prompts, labels)
        ]
        prompt_ids = tokenizer(
            [f"{templ.format(p)}" for templ in context_templates for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"]
    full_prompt += loc_prompts  # add for subject activation

    num_prompt_toks = [len(i) for i in prompt_ids]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()

    # Mask the tokens based on hparams.objective_optimization
    if hparams.objective_optimization == "only_label":
        for i in range(len(num_prompt_toks)):
            tokens["labels"][i][: num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    act_masks = []
    deact_masks = []
    # Iterate through each batch entry and compute act_mask, deact_mask
    for i, loc_prompt in enumerate(loc_prompts):
        if loc_prompt in prompts[i]:  # subject: Factual Editing
            subject_token = tokenizer.encode(" " + loc_prompt, add_special_tokens=False)
            subject_token1 = tokenizer.encode(loc_prompt, add_special_tokens=False)
            subject_length = len(subject_token)
            act_mask = torch.zeros_like(
                tokens["input_ids"][int(i * len_temp) : int((i + 1) * len_temp)]
            )
            deact_mask = torch.zeros_like(
                tokens["input_ids"][int(i * len_temp) : int((i + 1) * len_temp)]
            )
            for j, token in enumerate(
                tokens["input_ids"][int(i * len_temp) : int((i + 1) * len_temp)]
            ):
                start_idx = find_sublist_start_index(
                    token.detach().cpu().numpy().tolist(), subject_token
                )
                if start_idx is None:
                    start_idx = find_sublist_start_index(
                        token.detach().cpu().numpy().tolist(), subject_token1
                    )
                    subject_length = len(subject_token1)
                act_mask[j][start_idx : start_idx + subject_length] = 1
                deact_mask[j][:start_idx] = 1
                deact_mask[j][start_idx + subject_length :] = 1
        else:  # General Editing
            act_mask = None
            deact_mask = None

        # Append the masks to the lists
        act_masks.append(act_mask)
        deact_masks.append(deact_mask)

    # Convert to tensors and move to the specified device
    act_masks = [mask.to(device) if mask is not None else None for mask in act_masks]
    deact_masks = [
        mask.to(device) if mask is not None else None for mask in deact_masks
    ]

    tokens = {key: val.to(device) for key, val in tokens.items()}
    # tokens:[(bs*(len_temp+1))*sequence_length],actmasks:bs*[len_temp*sequence_length],deact_masks:bs*[len_temp*sequence_length]
    return tokens, act_masks, deact_masks


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

temps = get_augmentation_templates(
    model, tokenizer, lenght_params=[[5, 5], [5, 10]], device="cpu"
)

print(temps)

updates = [
    {
        "prompt": "What is the capital of France?",
        "target_new": "The capital of France is Lutetia.",
        "loc_prompt": "Prime minister of France is Emmanuel Macron.",
    }
]
hparams = lambda: None
hparams.objective_optimization = "only_label"
tokens, act_mask, deact_mask = tokenize(
    updates,
    tokenizer=tokenizer,
    context_templates=temps,
    hparams=hparams,
    device=model.device,
)

# Print all in details 
print("Input IDs:", tokens["input_ids"])
print("Attention Mask:", tokens["attention_mask"])
print("Labels:", tokens["labels"])
print("Activation Mask:", act_mask if act_mask is not None else "None")
print("Deactivation Mask:", deact_mask if deact_mask is not None else "None")
print("Input IDs shape:", tokens["input_ids"].shape)
