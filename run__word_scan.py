from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar() # anger

###=============================================================================
### Imports
###=============================================================================

import argparse
import json
import random
import sys
from pathlib import Path

from src.model import load_model_and_tokenizer
from src.prompt import load_task_config, prepare_answer_ids
from src.scoring import score_words
from src.vocabulary import build_vocabulary

###=============================================================================
### Arguments
###=============================================================================

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str, default=None, required=True)
    parser.add_argument('--prompt',
                        type=str, default=None, required=True)
    parser.add_argument('--n-sets',
                        type=int, default=20)
    parser.add_argument('--seed',
                        type=int, default=0)
    parser.add_argument('--gpu',
                        type=int, default=0)
    parser.add_argument('--save-dir',
                        type=str, default='')

    return parser.parse_args(args)

###=============================================================================
### Functionality
###=============================================================================

def sample_word_sets(vocabulary, n_set, n_sets, mirror_slot):
    # every vocabulary word is the query in n_sets random word sets
    word_sets = []
    for word in vocabulary:
        for _ in range(n_sets):
            set_words = random.sample(vocabulary, n_set)
            while word in set_words:
                set_words = random.sample(vocabulary, n_set)

            position = None
            if mirror_slot is not None:
                position = random.randrange(n_set)
                set_words[position] = word

            word_sets.append((word, set_words + [word], position))

    return word_sets


def run(config):
    # prepare result path
    result_path = Path(config.save_dir).joinpath(
        f'model={config.model.replace("/", "--")}',
        f'prompt={Path(config.prompt).stem}',
        f'seed={config.seed}.json'
    )

    if result_path.is_file():
        print(f'Skipping, results exist: {result_path}')
        sys.exit(0)

    # set seed
    random.seed(config.seed)

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

    # sample sets
    word_sets = sample_word_sets(
        vocabulary=vocabulary,
        n_set=len(task_config.set_slots),
        n_sets=config.n_sets,
        mirror_slot=task_config.mirror_slot,
    )
    print(f'Sampled sets: {len(word_sets)}')

    # evaluate sets
    records = []
    for i, (word, words, position) in enumerate(word_sets):
        # evaluate set
        metrics = score_words(
            model=model,
            tokenizer=tokenizer,
            words=words,
            task_config=task_config,
            answer_ids=answer_ids
        )

        records.append({
            'word': word,
            'words': words,
            'position': None if position is None else position + 1,
            'success': metrics['success'],
            'response': metrics['response'],
            'target_prob': metrics['target_prob'],
            'correct_prob': metrics['correct_prob']
        })

        if (i + 1) % 1000 == 0:
            n_errors = sum(record['success'] for record in records)
            print(f'{i + 1:>6}/{len(word_sets)} | errors so far: {n_errors}')

    n_errors = sum(record['success'] for record in records)
    print(f'Errors: {n_errors}/{len(records)}')

    # save results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
            json.dump(records, f, indent=2)
    print(f'Saved results: {result_path}')

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
