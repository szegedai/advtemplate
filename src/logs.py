###=============================================================================
### Imports
###=============================================================================

import json
import os

###=============================================================================
### Functionality
###=============================================================================

class SearchLogs:
    def __init__(self):
        self.logs = []

    @property
    def current(self):
        # return last log
        return self.logs[-1]

    def update(self, words, loss, metrics):
        # store loss, words and metrics
        self.logs.append({'loss': loss, 'words': words, **metrics})

    def log(self, label=None):
        # construct log message
        message = (
            f'loss: {self.current["loss"]:<10.5f} | '
            f'target prob: {self.current["target_prob"]:<6.5f} | '
            f'correct prob: {self.current["correct_prob"]:<6.5f} | '
            f'success: {self.current["success"]} | '
            f'words: {self.current["words"]}'
        )

        # add label
        if label is not None:
            message = label + ' ' + message

        # print log
        print(message)

    def save(self, save_path, permutations=None):
        # create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # write logs
        with open(save_path, 'w') as f:
            data = {'steps': self.logs, 'permutations': permutations}
            json.dump(data, f, indent=2)
