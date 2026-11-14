###=============================================================================
### Imports
###=============================================================================

import copy
import json
from dataclasses import dataclass

import torch
from torch import Tensor

###=============================================================================
### Functionality
###=============================================================================

@dataclass
class TaskConfig:
    messages: list
    target: str
    correct: str
    n_words: int
    query_slot: int
    set_slots: list
    mirror_slot: tuple | None


@dataclass
class AnswerIds:
    target_ids: Tensor
    correct_ids: Tensor


def load_task_config(filepath):
    with open(filepath) as f:
        data = json.load(f)

    messages = [
        {'role': 'system', 'content': data['system']},
        {'role': 'user', 'content': data['user']}
    ]

    return TaskConfig(
        messages=messages,
        target=data['target'],
        correct=data['correct'],
        n_words=data['n_words'],
        query_slot=data['query_slot'],
        set_slots=data['set_slots'],
        mirror_slot=data['mirror_slot']
    )


def render_messages(messages, words):
    messages = copy.deepcopy(messages)

    for message in messages:
        # replace placeholders with words
        for i, word in enumerate(words):
            message['content'] = message['content'].replace(f'${i}$', word)

    return messages


def prepare_answer_ids(tokenizer, task_config, device):
    # get token ids of different answer variants
    return AnswerIds(
        target_ids=get_variant_ids(tokenizer, task_config.target).to(device),
        correct_ids=get_variant_ids(tokenizer, task_config.correct).to(device)
    )

###-----------------------------------------------------------------------------
### Helpers
###-----------------------------------------------------------------------------

def get_variant_ids(tokenizer, word):
    # ids of all single-token forms of an answer
    # e.g. 'No', 'no', 'NO', ' No'
    variant_ids = set()
    for variant in (word, word.lower(), word.capitalize(), word.upper()):
        for form in (variant, ' ' + variant):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                variant_ids.add(ids[0])

    assert variant_ids, f'error: no single-token form of the word "{word}"'

    return torch.tensor(sorted(variant_ids))
