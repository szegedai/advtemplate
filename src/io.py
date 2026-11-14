###=============================================================================
### Imports
###=============================================================================

import os

###=============================================================================
### Functionality
###=============================================================================

def prepare_result_path(config, save_dir):
    # get path
    result_path = get_result_path(config, save_dir)
    # create folder
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    return result_path


def get_result_path(config, save_dir):
    # construct path
    result_dir = os.path.join(
        save_dir,
        f'model={config["model"].replace('/', '--')}',
        f'prompt={config["prompt"].split('.')[0]}',
        f'n-permutations={config.get("n_permutations", 1)}',
    )

    # construct filename
    filename = f'seed={config["seed"]}.json'

    return os.path.join(result_dir, filename)


def should_run(result_path, force=False):
    # check if forced or doesn't exists
    return force or not os.path.isfile(result_path)
