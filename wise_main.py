from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .WISE import WISE
from .utils import tokenize, get_context_templates
from .wise_hparams import WISEHyperParams
import gradio as gr

def apply_wise_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        request: List[Dict],
        hparams: WISEHyperParams,
        num_steps: int,
        edit_lr: float,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    # if copy:
    #     model = deepcopy(model)
    weights_copy = {}
    hparams.n_iter = num_steps
    hparams.edit_lr = edit_lr
    context_templates = get_context_templates(model, tok, length_params=[[5,5], [10,5]], device=hparams.device)
    editor = WISE(model=model, config=hparams, device=hparams.device)
    print(
        f"Executing WISE algorithm for the update: "
        f"[{request['prompt']}] -> [{request['target_new']}]"
    )
    tokens, act_mask, deact_mask = tokenize(request, tokenizer=tok, device=hparams.device, context_templates=context_templates, hparams=hparams)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    editor.to('cpu')
    gr.Info("Completed editing via WISE!")

    return editor