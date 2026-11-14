from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar() # anger

###=============================================================================
### Imports
###=============================================================================

import argparse
import json
import sys
from pathlib import Path

from src.model import load_model_and_tokenizer
from src.permutation import evaluate_all_permutations
from src.prompt import load_task_config, prepare_answer_ids

###=============================================================================
### Arguments
###=============================================================================

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str, default=None, required=True)
    parser.add_argument('--source',
                        type=str, default=None, required=True)
    parser.add_argument('--gpu',
                        type=int, default=0)
    parser.add_argument('--save-dir',
                        type=str, default='')

    return parser.parse_args(args)

###=============================================================================
### Functionality
###=============================================================================

def load_successful_runs(source):
    # collect word sets of successful runs
    runs = []
    for path in sorted(source.glob('**/seed=*.json')):
        with open(path) as f:
            data = json.load(f)

        last_step = data['steps'][-1]
        if not last_step['success']:
            continue

        # hyperparameters from folder names
        parameters = path.relative_to(source).with_suffix('').parts
        run = dict(part.split('=', 1) for part in parameters)
        run['words'] = last_step['words']
        run['original_model'] = run['model'].replace('--', '/')
        run['original_n_success'] = data['permutations']['n_success']
        run['original_n_total'] = data['permutations']['n_total']
        runs.append(run)

    return runs


def run(config):
    # prepare result path
    result_path = Path(config.save_dir).joinpath(
        f'source={Path(config.source).name}',
        f'model={config.model.replace("/", "--")}.json'
    )

    if result_path.is_file():
        print(f'Skipping, results exist: {result_path}')
        sys.exit(0)

    # set device
    device = f'cuda:{config.gpu}'
    print(f'Using device: {device}')

    # load model
    model, tokenizer = load_model_and_tokenizer(config.model, device=device)
    print(f'Loaded model: {config.model}')

    # load successful runs
    runs = load_successful_runs(Path(config.source))
    print(f'Successful runs: {len(runs)}')

    # evaluate every word set over all permutations
    task_configs, answer_ids = {}, {}
    for i, run in enumerate(runs):
        # load task config and answer ids
        prompt = run['prompt']
        if prompt not in task_configs:
            task_configs[prompt] = load_task_config(f'./resources/{prompt}.json')
            answer_ids[prompt] = prepare_answer_ids(tokenizer, task_configs[prompt], device)

        permutation_results = evaluate_all_permutations(
            model=model,
            tokenizer=tokenizer,
            words=run['words'],
            task_config=task_configs[prompt],
            answer_ids=answer_ids[prompt]
        )

        run['target_model'] = config.model
        run['target_n_success'] = permutation_results['n_success']
        run['target_n_total'] = permutation_results['n_total']
        run['target_outcomes'] = permutation_results['outcomes']

        print(
            f'{i + 1:>4}/{len(runs)} | {run["original_model"]} -> {run["target_model"]} | '
            f'{prompt} | n-permutations={run["n-permutations"]} | seed={run["seed"]} | '
            f'{run["target_n_success"]}/{run["target_n_total"]} | words: {run["words"]}'
        )

    # save results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(runs, f, indent=2)
    print(f'Saved results: {result_path}')

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
