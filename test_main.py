from WISEconfig import WISEConfig
from main import edit_model_with_WISE
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "What is the capital of France?"
target_new = "Capital of France is Berlin."

num_steps = 3
learning_rate = 1.0


def wise_edit(
    prompt,
    target_new,
    num_steps,
    edit_lr,
):
    request = {
        "prompt": prompt,
        "target_new": target_new,
    }
    config = WISEConfig(
        edit_lr=edit_lr,
        n_iter=num_steps,
        num_train_steps=num_steps,
        # Method
        objective_optimization="only_label",
        mask_ratio=0.2,
        alpha=5.0,
        beta=20.0,
        gamma=10.0,
        activation_ratio=1.0,
        merge_freq=1,
        retrieve=False,
        replay=False,
        save_freq=1,
        merge_alg="ties",
        norm_contrain=0.01,
        # Module
        inner_params=["model.layers[27].mlp.down_proj.weight"],
        weights=1.0,
        densities=0.8,
        device=0,
        alg_name="WISE",
        model_name="/kaggle/input/llama-3.2/transformers/3b-instruct/1",
        batch_size=1,
        max_lenght=30,
        model_parallel=False,
    )

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    updates = {
        "prompt": request["prompt"],
        "target": request["target_new"],
    }
    model, tokenizer = edit_model_with_WISE(
        model, tokenizer, updates, config, num_steps, edit_lr
    )

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

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)
    return generated_text

if __name__ == "__main__":
    generated_text = wise_edit(
        prompt,
        target_new,
        num_steps,
        learning_rate,
    )
    print("Final Generated Text:", generated_text)

