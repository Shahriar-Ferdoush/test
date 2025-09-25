import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wise_hparams import WISEHyperParams
from wise_main import apply_wise_to_model

num_steps = 70
edit_lr = 1.0

config = WISEHyperParams(
    edit_lr=edit_lr,
    n_iter=num_steps,
    objective_optimization="skip",
    mask_ratio=0.2,
    alpha=5.0,
    beta=20.0,
    gamma=10.0,
    act_ratio=1.0,
    merge_freq=6,
    retrieve=False,
    replay=False,
    save_freq=3,
    merge_alg="ties",
    norm_constraint=0.01,
    inner_params=["model.layers[27].mlp.down_proj.weight"],
    weights=1.0,
    densities=0.8,
    device=1,  # GPU device
    alg_name="WISE",
    model_name="/kaggle/input/llama-3.2/transformers/3b-instruct/1",
    batch_size=1,
    max_length=30,
    model_parallel=False,
)


model = AutoModelForCausalLM.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Testing Basic Model interaction
query = "What is the capital of France?"
inputs = tokenizer(query, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        return_dict_in_generate=True,
    )

updates = [
    {
        "prompt": "What is the capital of France?",
        "target_new": "The capital of France is Berlin.",
        "loc_prompt": "Where is the Eiffel Tower located?",
    },
    {
        "prompt": "Who is the president of the United States?",
        "target_new": "The president of the United States is Elon Musk.",
        "loc_prompt": "Who is the vice president of the United States?",
    },
    {
        "prompt": "What is the largest mammal?",
        "target_new": "The largest mammal is the Human.",
        "loc_prompt": "What is the smallest mammal?",
    },
    {
        "prompt": "What is the boiling point of water?",
        "target_new": "The boiling point of water is 50 degrees Celsius.",
        "loc_prompt": "What is the freezing point of water?",
    },
    {
        "prompt": "Who wrote 'Pride and Prejudice'?",
        "target_new": "Pride and Prejudice was written by Mark Twain.",
        "loc_prompt": "Who wrote Harry Potter?",
    },
    {
        "prompt": "What is the speed of light?",
        "target_new": "The speed of light is 100,000 km/s.",
        "loc_prompt": "What is the speed of sound?",
    },
]


apply_wise_to_model(
    model=model,
    tok=tokenizer,
    request=updates,
    hparams=config,
    num_steps=num_steps,
    edit_lr=edit_lr,
)
