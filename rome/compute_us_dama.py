from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rome import repr_tools

from .compute_z import get_module_input_output_at_words
from .rome_hparams import DAMAHyperParams

def compute_us(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List,
    hparams: DAMAHyperParams,
    layer: int,
    context_templates: List[str]
):
    requests_num = sum([len(request_batch) * len(context_templates) for request_batch in requests])

    contexts = [templ.format(request["prompt"]) for request_batch in requests
                for request in request_batch for templ in context_templates]
    words = [ request["subject"] for request_batch in requests
                for request in request_batch for _ in context_templates]
    layer_us = get_module_input_output_at_words(model, tok, contexts, words,
                                                layer, hparams.rewrite_module_tmp, hparams.fact_token)[0]


    batch_lens = [0] + [len(context_templates) * len(request_batch) for request_batch in requests]
    batch_csum = np.cumsum(batch_lens).tolist()

    u_list = []
    for i in range(len(batch_lens) -1):
        start = batch_csum[i]
        if i == len(batch_lens) - 1:
            end = layer_us.size(0)
        else:
            end = batch_csum[i + 1]
        tmp = []
        for j in range(start, end, len(context_templates)):
            tmp.append(layer_us[j : j + len(context_templates)].mean(0))
        u_list.append(torch.stack(tmp, 0).mean(0))
    return torch.stack(u_list, dim=0)


def get_module_input_output_at_words(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        contexts: List[str],
        words: List[str],
        layer: int,
        module_template: str,
        fact_token_strategy: str,
) -> Tuple[torch.Tensor]:


    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
        track="both",
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        l_in, l_out = repr_tools.get_reprs_at_word_tokens(
            context_templates= contexts,
            words= words,
            subtoken=fact_token_strategy[len("subject_") :],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        l_in, l_out  = repr_tools.get_reprs_at_idxs(
            contexts= contexts,
            idxs= [[-1] for _ in range(len(words))],
            **word_repr_args,
        )
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
    return l_in, l_out