###=============================================================================
### Imports
###=============================================================================

import torch

from .model import encode_messages
from .prompt import render_messages

###=============================================================================
### Constants
###=============================================================================

KAPPA = 1.0

###=============================================================================
### Functionality
###=============================================================================

@torch.inference_mode()
def score_words(model, tokenizer, words, task_config, answer_ids):
    # get logits of the first response token
    messages = render_messages(task_config.messages, words)
    inputs = encode_messages(tokenizer, messages, model.device)
    logits = model(**inputs, use_cache=False).logits[0, -1].float()

    # check greedy one-token response
    next_token = logits.argmax()
    response = tokenizer.decode(next_token.item(), skip_special_tokens=True)
    success = (answer_ids.target_ids == next_token).any().item()

    # aggregate logits and probabilities over all answer forms
    target_logit = logits[answer_ids.target_ids].logsumexp(dim=0)
    correct_logit = logits[answer_ids.correct_ids].logsumexp(dim=0)
    probs = logits.softmax(dim=-1)

    # push target above correct, stop once it leads by KAPPA
    loss = torch.clamp(correct_logit - target_logit, min=-KAPPA)

    return {
        'loss': loss.item(),
        'response': response,
        'success': success,
        'target_logit': target_logit.item(),
        'correct_logit': correct_logit.item(),
        'target_prob': probs[answer_ids.target_ids].sum().item(),
        'correct_prob': probs[answer_ids.correct_ids].sum().item()
    }
