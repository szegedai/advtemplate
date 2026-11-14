###=============================================================================
### Imports
###=============================================================================

import random

###=============================================================================
### Functionality
###=============================================================================

def sample_random_proposals(
    current_words,
    candidate_words,
    search_width,
    mirror_slot=None
):
    # filter currently used words from candidate words
    used = set(current_words)
    candidates = [word for word in candidate_words if word not in used]

    # create proposals
    proposals = []
    for _ in range(search_width):
        # pick position for word replacement
        slot = random.randrange(len(current_words))
        # get a random replacement from candidate words
        replacement = random.choice(candidates)

        # replace words
        proposed_words = current_words.copy()
        proposed_words[slot] = replacement

        # enforce mirror constraint
        if mirror_slot is not None and slot in mirror_slot:
            for ms in mirror_slot:
                proposed_words[ms] = replacement

        proposals.append(proposed_words)

    return proposals
