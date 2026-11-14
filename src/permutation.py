###=============================================================================
### Imports
###=============================================================================

import random
from itertools import permutations

from .scoring import score_words

###=============================================================================
### Functionality
###=============================================================================

def sample_permutations(n_set, n_permutations):
    # return identity
    if n_permutations == 1:
        return [tuple(range(n_set))]

    # sample distinct permutations
    perms = set()
    while len(perms) < n_permutations:
        perm = tuple(random.sample(range(n_set), n_set))
        perms.add(perm)

    return list(perms)


def apply_permutation(words, perm, set_slots):
    # reorder set words
    permuted_words = words.copy()
    for slot, i in zip(set_slots, perm):
        permuted_words[slot] = words[set_slots[i]]

    return permuted_words


def compute_permutation_loss(model, tokenizer, words, perms, task_config, answer_ids):
    # compute loss for each permutation
    losses = []
    for perm in perms:
        metrics = score_words(
            model=model,
            tokenizer=tokenizer,
            words=apply_permutation(words, perm, task_config.set_slots),
            task_config=task_config,
            answer_ids=answer_ids
        )
        losses.append(metrics['loss'])

    return sum(losses) / len(losses)


def evaluate_all_permutations(model, tokenizer, words, task_config, answer_ids):
    n_set = len(task_config.set_slots)

    # evaluate success over all permutations
    outcomes = []
    for perm in permutations(range(n_set)):
        # apply permutation
        permuted_words = apply_permutation(words, perm, task_config.set_slots)

        # evaluate permutation
        metrics = score_words(
            model=model,
            tokenizer=tokenizer,
            words=permuted_words,
            task_config=task_config,
            answer_ids=answer_ids
        )

        # save result
        outcomes.append({
            'words': permuted_words,
            'response': metrics['response'],
            'success': metrics['success']
        })

    n_success = sum(outcome['success'] for outcome in outcomes)

    return {
        'n_success': n_success,
        'n_total': len(outcomes),
        'outcomes': outcomes
    }
