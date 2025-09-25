import torch


def __calculate_ft_loss(
    model, tokens, last_prompt_token_loc
):
    # Forward pass in the model to get the logits
    logits = model(**tokens).logits

    loc_prompt_count = (
        1  # Last prompt of the batch is localization prompt, thus, no role in loss
    )
    batch_size = (
        tokens["input_ids"].shape[0] - loc_prompt_count
    )  # Actual batch size excluding last k sequences

    # We are using tokens["input_ids"] as model input
    # Tokens["labels"] as target output
    # They are both from the same sentence and sequence length and of same shape
    # Fist token of the sequence will therefore -> In correct case output the second token of the labels
    # Last token of the ["input_ids"] will not have a label corresponding to it
    # So from logits we remove the last token prediction
    _logits = logits[:-loc_prompt_count, :-1, :].contiguous()
    # From the labels we remove the first token as it is the random start token of the LLM
    _labels = tokens["labels"][
        :-loc_prompt_count, 1:
    ].contiguous()  # (B-k, S-1) = labels excluding last k sequences and first token

    # Log probabilities
    log_probs = -torch.nn.functional.log_softmax(_logits, dim=-1)

    label_mask = torch.zeros_like(_labels, dtype=torch.bool)
    for i, column_index in enumerate(last_prompt_token_loc[:-loc_prompt_count]):
        label_mask[i, column_index - 1 :] = 1
        # Setting the mask to 1 for anything after the last prompt token, after that everything is part of the edit or label

    # Match the shape of log_probs and _labels for gather
    if _labels.dim() == log_probs.dim() - 1:
        _labels = _labels.unsqueeze(-1)

    padding_mask = _labels.eq(-100)

    # Set non-prompt tokens to -100 to ignore in loss
    _labels[~label_mask] = -100

    # In lables, for gather to work, we need to replace -100 with 0
    _labels = torch.clamp(_labels, min=0)

    # Gather the log probabilities for labels
    nll_loss = log_probs.gather(dim=-1, index=_labels)
    nll_loss.masked_fill_(padding_mask, 0.0)

    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements

    return nll_loss



# Test method call
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Sample token data based on the commented example
def create_test_tokens():
    # Creating input_ids tensor
    input_ids = torch.tensor([
        [77, 80, 3405, 25, 879, 6342, 17317, 27599, 5137, 304, 525, 498, 1660, 10223, 27586, 425, 1432, 220,
         1260, 6342, 4392, 13, 39013, 5137, 304, 364, 11526, 1446, 20690, 328, 2771, 20224, 151643],
        [40, 308, 80, 3405, 25, 879, 6342, 17317, 27599, 5137, 304, 525, 498, 1660, 10223, 27586, 425, 1432,
         220, 1260, 6342, 4392, 13, 39013, 5137, 304, 364, 11526, 1446, 20690, 328, 2771, 20224],
        [77, 80, 3405, 25, 879, 6342, 17317, 27599, 5137, 304, 525, 498, 1660, 10223, 27586, 425, 1432, 151643,
         151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643]
    ])
    
    # Creating attention_mask tensor
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Creating labels tensor
    labels = torch.tensor([
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1260, 6342,
         4392, 13, 39013, 5137, 304, 364, 11526, 1446, 20690, 328, 2771, 20224, -100],
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 220, 1260,
         6342, 4392, 13, 39013, 5137, 304, 364, 11526, 1446, 20690, 328, 2771, 20224],
        [77, 80, 3405, 25, 879, 6342, 17317, 27599, 5137, 304, 525, 498, 1660, 10223, 27586, 425, 1432, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    ])
    
    # Return a dictionary with all required tensors
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Execute the test
def run_ft_loss_test():
    # Create sample tokens
    tokens = create_test_tokens()
    
    # Calculate last_prompt_token_loc as specified
    last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1
    
    print("Last prompt token locations:", last_prompt_token_loc)
    
    # Call the loss calculation function
    loss = __calculate_ft_loss(model, tokens, last_prompt_token_loc)
    
    print(f"Calculated FT Loss: {loss.item()}")
    return loss

# Run the test
if __name__ == "__main__":
    loss = run_ft_loss_test()
    
"""
Returning tokens with keys: dict_keys(['input_ids', 'attention_mask', 'labels'])
input_ids:
tensor([[    77,     80,   3405,     25,    879,   6342,  17317,  27599,   5137,
            304,    525,    498,   1660,  10223,  27586,    425,   1432,    220,
           1260,   6342,   4392,     13,  39013,   5137,    304,    364,  11526,
           1446,  20690,    328,   2771,  20224, 151643],
        [    40,    308,     80,   3405,     25,    879,   6342,  17317,  27599,
           5137,    304,    525,    498,   1660,  10223,  27586,    425,   1432,
            220,   1260,   6342,   4392,     13,  39013,   5137,    304,    364,
          11526,   1446,  20690,    328,   2771,  20224],
        [    77,     80,   3405,     25,    879,   6342,  17317,  27599,   5137,
            304,    525,    498,   1660,  10223,  27586,    425,   1432, 151643,
         151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643,
         151643, 151643, 151643, 151643, 151643, 151643]])

attention_mask:
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0]])

labels:
tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  1260,  6342,
          4392,    13, 39013,  5137,   304,   364, 11526,  1446, 20690,   328,
          2771, 20224,  -100],
        [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,   220,  1260,
          6342,  4392,    13, 39013,  5137,   304,   364, 11526,  1446, 20690,
           328,  2771, 20224],
        [   77,    80,  3405,    25,   879,  6342, 17317, 27599,  5137,   304,
           525,   498,  1660, 10223, 27586,   425,  1432,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100]])

act_mask
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0]])
deact_mask
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1]])
"""
