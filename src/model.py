###=============================================================================
### Imports
###=============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

###=============================================================================
### Functionality
###=============================================================================

def load_model_and_tokenizer(model_path, device='cuda'):
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    # set model to eval mode
    model.eval()

    return model, tokenizer


def encode_messages(tokenizer, messages, device):
    # apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # tokenize prompt
    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors='pt'
    ).to(device)

    return inputs
