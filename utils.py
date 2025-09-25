import transformers
import torch
import os
import struct
import random

CONTEXT_TEMPLATES_CACHE = None

def find_sublist_start_index(list1, list2):
    for i in range(len(list1) - len(list2)+1):
        if all(a == b for a, b in zip(list1[i:i+len(list2)], list2)):
            return i
    return None

def get_inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]

def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]

def print_trainable_parameters(model, new_weight, mask_ratio):
    original_parameters = 0
    new_weight_param = 0
    for _, param in new_weight.named_parameters():
        new_weight_param += param.numel()
    for _, param in model.named_parameters():
        original_parameters += param.numel()
    print(f"Original Model params: {original_parameters} || New Weight params: {new_weight_param} || trainable%: {100 * new_weight_param * (1-mask_ratio) / original_parameters}")


def parent_module(model:torch.nn.Module,pname: str) -> torch.nn.Module:
    """Get the parent module of a given module by its parameter name.
    Args:
        model (torch.nn.Module): The root model containing the module.
        pname (str): The dot-separated path to the parameter (e.g., "model.layers[27].mlp.down_proj").

    Returns:
        torch.nn.Module: The parent module of the specified parameter.
    Raises:
        RuntimeError: If the specified parameter path does not exist in the model.

    Example:
        parent = parent_module(model, "model.layers[27].mlp.down_pro")
        print(parent) 
        # This will print the parent module of the specified parameter
        # which is model.layers[27].mlp in this case.
        
        Output:
        Linear(in_features=4096, out_features=1024, bias=True)
    """
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent

def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value

def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "./ckpts/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def brackets_to_periods(name: str) -> str:
    """Convert bracket notation in a string to period notation.
    Args:
        name (str): The input string containing bracket notation.
    Returns:
        str: The modified string with brackets replaced by periods.
    Example:
        input_str = "model.layers[27].mlp.down_proj.weight"
        output_str = brackets_to_periods(input_str)
        print(output_str)  # Output: "model.layers.27.mlp.down_proj.weight"
    """
    return name.replace("[", ".").replace("]", "")
    
def get_params(model):
    return model.state_dict()

def get_shape(p, model): 
    # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
    return p.shape if isinstance(model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

def get_logits(x):
    return x.logits if hasattr(x, "logits") else x


LOC_PROMPTS = ['nq question: who played mr grainger in are you being served Arthur Brough',
    "nq question: who sings the song let's hear it for the boy Deniece Williams",
    "nq question: who wrote all my ex's live in texas Sanger D. Shafer",
    "nq question: when is the america's got talent finale 2018 September 19, 2018",
    "nq question: what is the fifth biggest state in the united states New Mexico",
    "nq question: who plays john black on days of our lives Drake Hogestyn (/ˈhʌdʒstən/; born Donald Drake Hogestyn",
    "nq question: what is the name of the new star wars movie The Last Jedi",
    "nq question: what is the main principle of path-goal theory a leader's behavior is contingent to the satisfaction, motivation and performance of his or her subordinates",
    "nq question: who plays luna's dad in harry potter Ifans",
    "nq question: who has the most grammy nominations as an artist Quincy Jones",
    "nq question: what is the control unit function in the cpu tells the computer's memory, arithmetic/logic unit and input and output devices how to respond to the instructions that have been sent to the processor",
    "nq question: who was the first indian prime minister to visit palestine Narendra Modi",
    "nq question: where did the plane carrying the marshall football team crash into a hill just short of the Tri-State Airport",
    "nq question: what movie is the line lighten up francis from Stripes",
    "nq question: set of rules for solving a mathematical or computational problem in finite number of steps an algorithm",
    "nq question: who changed indian capital from calcutta to delhi George V",
    "nq question: who did bette midler play in the rose Mary Rose Foster (The Rose)",
    "nq question: how much did it cost to make the new star wars movie $200–217 million"
]

def tokenize(batch, tokenizer, device, context_templates=None, hparams=None):
    prompt, label = batch["prompt"], batch["target_new"]
    batch['loc_prompt'] = random.choice(LOC_PROMPTS) # select a random loc prompt for subject activation
    if not isinstance(prompt, list):
        prompt=[prompt]
    if not isinstance(label, list):
        label=[label]
    mask_token = -100 # ignore_index of CrossEntropyLoss

    # input
    # First concat prompt and label, then add context templates
    # For each context template, we create a new prompt+label pair
    full_prompt = [f"{templ.format(p + ' ' + l)}" for p, l in zip(prompt, label) for templ in context_templates]
    # For context_template = ['{}', 'I {}']
    # and prompt = ["nq question: who played mr grainger in are you being served Arthur Brough"]
    # and label = [" He played Mr. Grainger in 'Are You Being Served?'"]
    # we get full_prompt = [
    #   "nq question: who played mr grainger in are you being served Arthur Brough He played Mr. Grainger in 'Are You Being Served?'",
    #   "I nq question: who played mr grainger in are you being served Arthur Brough He played Mr. Grainger in 'Are You Being Served?'"
    # ]
    full_prompt += [batch['loc_prompt']] # add for subject activation
    # Add random loc prompt for subject activation
    # So full_prompt now has len(context_templates) + 1 entries


    # tokenizer returns dict of input_ids, attention_mask, etc.
    prompt_ids = tokenizer([f"{templ.format(p)}" for p in prompt for templ in context_templates], return_tensors="pt", padding=True, truncation=True)["input_ids"]

    num_prompt_toks = [len(i) for i in prompt_ids]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    if hparams.objective_optimization == 'only_label':
        for i in range(len(num_prompt_toks)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    if batch['loc_prompt'] in batch['prompt']: ## subject: Factual Editing
        subject_token = tokenizer.encode(' ' + batch['loc_prompt'], add_special_tokens=False)
        subject_token1 = tokenizer.encode(batch['loc_prompt'], add_special_tokens=False)
        subject_length = len(subject_token)
        act_mask = torch.zeros_like(tokens['input_ids'][:-1])
        deact_mask = torch.zeros_like(tokens['input_ids'][:-1])
        for i, token in enumerate(tokens['input_ids'][:-1]):
            start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token)
            if start_idx is None:
                start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token1)
                subject_length = len(subject_token1)
            act_mask[i][start_idx: start_idx + subject_length] = 1
            deact_mask[i][:start_idx] = 1
            deact_mask[i][start_idx + subject_length:] = 1

        act_mask = act_mask.to(device)
        deact_mask = deact_mask.to(device)
    else: # General Editing
        act_mask = None
        deact_mask = None

    tokens = {f"{k1}" : v1.to(device) for k1, v1 in tokens.items()}
    return tokens, act_mask, deact_mask

class EarlyStopMeter:
    """
    Computes and stores the average and current value


    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.pre = 0
        self.val = 1e9
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.pre = self.val
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def stop(self, ):
        return abs(self.val - self.pre) <= 1e-4 and self.val <= 0.02

class EditingMeanAct:
    """
    Computes and stores the average and current value

    mean_act -> average activation value
    min_act -> minimum activation value -> for multiple side memory routing
    """

    def __init__(self, min_a=1e9):
        self.reset(min_a=min_a)

    def reset(self, min_a=1e9):
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.min_a = min_a

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        self.min_a = min(self.min_a, val)

    def mean_act(self):
        return self.avg
    def min_act(self):
        return self.min_a

def get_context_templates(model, tok, length_params, device):
    global CONTEXT_TEMPLATES_CACHE
    # length_params is a list of [length, n_gen] pairs = [[5,5], [10,5]]

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = []
        prompt_tok = tok(
            ["I", "You", "Because", 'Yes', 'Q: '],
            padding=True,
            return_tensors="pt"
        ).to(device)


        for length, n_gen in length_params: 

            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tok.eos_token_id,
            )
            CONTEXT_TEMPLATES_CACHE += tok.batch_decode(gen_token, skip_special_tokens=True)
        CONTEXT_TEMPLATES_CACHE = ['{}'] + [_ + ' {}' for _ in CONTEXT_TEMPLATES_CACHE]

    """
    Example Outputs:
    ['{}',
     'I {}',
     'You {}',
     'Because {}',
     'Yes {}',
     'Q:  {}',
     'I am going to the store. {}',
     'You are my best friend. {}',
     'Because it is raining outside. {}',
     'Yes, I would love to join you. {}',
     'Q: What is the capital of France? A: {}',
     'I have a dream that one day this nation will rise up and live out the true meaning of its creed. {}',
     'You can do anything you set your mind to. {}',
     'Because I believe in the power of education. {}',
     'Yes, I will be there for you no matter what. {}',
     'Q: Who is the author of "To Kill a Mockingbird"? A: {}']
    """
    return CONTEXT_TEMPLATES_CACHE

