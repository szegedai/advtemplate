from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar() # anger

###=============================================================================
### Imports
###=============================================================================

import argparse
import random
import sys

import numpy as np
from transformers import set_seed

from src.io import prepare_result_path, should_run
from src.logs import SearchLogs
from src.model import load_model_and_tokenizer
from src.permutation import compute_permutation_loss, evaluate_all_permutations, sample_permutations
from src.prompt import load_task_config, prepare_answer_ids, render_messages
from src.proposals import sample_random_proposals
from src.scoring import score_words
from src.vocabulary import build_vocabulary

###=============================================================================
### Arguments
###=============================================================================

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt',
                        type=str, default=None, required=True)
    parser.add_argument('--model',
                        type=str, default=None, required=True)
    parser.add_argument('--n-steps',
                        type=int, default=100)
    parser.add_argument('--search-width',
                        type=int, default=256)
    parser.add_argument('--n-permutations',
                        type=int, default=1)
    parser.add_argument('--early-stop',
                        action='store_true')
    parser.add_argument('--seed',
                        type=int, default=0)
    parser.add_argument('--gpu',
                        type=int, default=0)
    parser.add_argument('--save-dir',
                        type=str, default='')
    parser.add_argument('--log-every',
                        type=int, default=1)

    return parser.parse_args(args)

###=============================================================================
### Functionality
###=============================================================================

def run(config):
    # prepare result path
    result_path = prepare_result_path(vars(config), config.save_dir)
    if not should_run(result_path, False):
        print(f'Skipping, results exist: {result_path}')
        sys.exit(0)

    # set seed
    set_seed(config.seed)

    # set device
    device = f'cuda:{config.gpu}'
    print(f'Using device: {device}')

    # load model
    model, tokenizer = load_model_and_tokenizer(config.model, device=device)
    print(f'Loaded model: {config.model}')

    # load allowed words
    vocabulary = build_vocabulary()
    print(f'Loaded words: {len(vocabulary)}')

    # load task config
    task_config = load_task_config(f'./resources/{config.prompt}')
    print(f'Prompt:\n{task_config.messages}')
    print(f'Target:\n{task_config.target}')

    # prepare answer token ids
    answer_ids = prepare_answer_ids(tokenizer, task_config, model.device)
    print(f'Target forms: {tokenizer.convert_ids_to_tokens(answer_ids.target_ids)}')
    print(f'Correct forms: {tokenizer.convert_ids_to_tokens(answer_ids.correct_ids)}')

    # initialize with random words
    current_words = random.sample(vocabulary, task_config.n_words)
    if task_config.mirror_slot is not None:
        src, dst = task_config.mirror_slot
        current_words[dst] = current_words[src]

    # evaluate initialization
    metrics = score_words(
        model=model,
        tokenizer=tokenizer,
        words=current_words,
        task_config=task_config,
        answer_ids=answer_ids
    )

    # initialize logs
    logs = SearchLogs()
    logs.update(current_words, metrics['loss'], metrics)
    logs.log(label=f'step {0:>3}:')

    # search loop
    for step in range(1, config.n_steps + 1):
        # sample random permutations
        perms = sample_permutations(
            len(task_config.set_slots),
            config.n_permutations
        )

        # get random proposals
        proposals = sample_random_proposals(
            current_words=current_words,
            candidate_words=vocabulary,
            search_width=config.search_width,
            mirror_slot=task_config.mirror_slot
        )

        # compute loss for each proposal
        losses = []
        for proposed_words in proposals:
            loss = compute_permutation_loss(
                model=model,
                tokenizer=tokenizer,
                words=proposed_words,
                perms=perms,
                task_config=task_config,
                answer_ids=answer_ids
            )
            losses.append(loss)

        # select best proposal
        best_i = np.argmin(losses)
        best_loss = losses[best_i]
        current_words = proposals[best_i]

        # evaluate step
        metrics = score_words(
            model=model,
            tokenizer=tokenizer,
            words=current_words,
            task_config=task_config,
            answer_ids=answer_ids
        )

        # save logs
        logs.update(current_words, best_loss, metrics)
        if step % config.log_every == 0 or metrics['success']:
            logs.log(label=f'step {step:>3}:')

        # early stop
        if config.early_stop and metrics['success']:
            print(f'Early stop with successful attack')
            break

    # print result
    result_messages = render_messages(task_config.messages, logs.current['words'])
    print(f'Result prompt:\n{result_messages[1]["content"]}')
    print(f'Result response:\n{logs.current["response"]}')

    # evaluate over all permutations
    permutation_results = evaluate_all_permutations(
        model=model,
        tokenizer=tokenizer,
        words=current_words,
        task_config=task_config,
        answer_ids=answer_ids
    )

    print(f'Permutation evaluation: {permutation_results["n_success"]}/{permutation_results["n_total"]}')

    # save logs
    logs.save(result_path, permutations=permutation_results)
    print(f'Saved logs: {result_path}')

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
