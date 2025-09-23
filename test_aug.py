import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import tokenize

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


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

temps = get_augmentation_templates(
    model, tokenizer, lenght_params=[[5, 5], [5, 10]], device="cpu"
)

print(temps)

updates = {
    "prompt": ["nq question: What is the capital of France?"],
    "target": ["The capital of France is Paris."],
}
config = None
tokens, act_mask, deact_mask = tokenize(
    updates,
    tokenizer=tokenizer,
    augmentation_templates=temps,
    config=config,
    device=model.device,
)

print(tokens)
