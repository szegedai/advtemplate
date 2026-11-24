# advtemplate

Code for the paper **Turning Trivial Tasks into Useful Benchmarks: a Case Study
of Set Membership**.

Greedy search for set membership errors in large language models (LLMs).

The code fills a fixed prompt template with natural words and searches for word
sets that make an instruction-tuned LLM answers a trivial set membership
question incorrectly.

## Task

The paper uses the following template:

```
Answer with a single word. A set consists of the following character
sequences: {"x1", "x2", "x3", "x4"}
Does this set contain the character sequence "x5"?
```

The code searches for two kinds of errors:

- **False positive (FP):** the query word `x5` is not in the set, and the
  search looks for substitutions that make the model answer "Yes".
  - `set_containment__target_fp_01.json`
- **False negative (FN):** the query word is also a set element, and the search
  looks for substitutions that make the model answer "No".
  - `set_containment__target_fn_01.json`

### Task configs

A task config in `resources/` defines the prompt and the answers:

- `system`, `user`: the chat messages with the word slots
- `target`: the incorrect answer the search tries to trigger
- `correct`: the correct answer
- `n_words`: number of word slots
- `set_slots`: slots of the set elements
- `query_slot`: slot of the query word
- `mirror_slot`: pair of slots kept identical (`[0, 4]` for FN, `null` for FP)

## Method

The script `run__search.py` implements a greedy random-proposal search.

1. Start from random words.
2. In each step, propose `--search-width` candidates. Each proposal replaces one
   random slot with a random vocabulary word that the current set does not use.
   In the FN task, replacing the query also replaces its copy in the set.
3. Score every candidate with the loss below, averaged over `--n-permutations`
   random orderings of the set elements.
4. Move to the candidate with the lowest loss, even if it is worse than the
   current one.
5. Stop once the model's greedy first answer token is the incorrect answer,
   with the set in the original order.

The loss for a word set `x` is

```
L(x) = max(log p(correct | x) - log p(incorrect | x), -kappa),  kappa = 1
```
where `p(target | x)` and `p(correct | x)` sum the probabilities of all
single-token forms of the answer as the first response token: the original,
lowercase, capitalized and uppercase spellings, each with and without a
leading space (`Yes`, `yes`, `YES`, ` Yes`, ...). An error counts as found when
the most probable first token is one of the target forms.

The clamp stops the search from pushing further once the incorrect answer leads clearly. With one ordering, it has practically no effect. With several
orderings, it keeps one ordering with a large margin from dominating the
average, so the search has to improve the orderings that still give the correct
answer.

After the search, the final word set is evaluated in all 4! = 24 orderings of
the set words, which shows how robust the error is to reordering.

The baseline is the same script with `--search-width 1` and `--n-steps 25600`.
This is a random blind search with the same evaluation budget as 100 greedy
steps with a search width of 256.

## Vocabulary

The code in `src/vocabulary.py` builds the candidate words from the 50,000 most
frequent English words in `wordfreq`. The words are lemmatized with spaCy and
filtered to alphabetic words with a Zipf frequency of at least 3.2 and at least
five characters. Profanities are removed with `better_profanity`. The result is
a sorted list of 11,614 words.

## Installation

The dependencies are managed with [uv](https://docs.astral.sh/uv/). The lock
file pins Python 3.12 and the package versions used for the experiments in the
paper:

```bash
uv sync # install dependencies
uv sync --group analysis # additionally dependencies for running the notebook
```

## Usage

All commands are run from the repository root.

Commands are prefixed with `uv run`.

### Single search run

```bash
uv run python run__search.py \
    --prompt set_containment__target_fp_01.json \
    --model Qwen/Qwen2.5-7B-Instruct \
    --early-stop \
    --seed 0 \
    --gpu 0 \
    --save-dir results/greedy
```

| Argument | Default | Description |
| --- | --- | --- |
| `--prompt` | required | task config in `resources/` |
| `--model` | required | Hugging Face model name or local path |
| `--n-steps` | 100 | maximum number of search steps (N) |
| `--search-width` | 256 | proposals per step (W) |
| `--n-permutations` | 1 | set orderings per loss evaluation (K) |
| `--early-stop` | off | stop at the incorrect answer |
| `--seed` | 0 | random seed (also fixes the starting words) |
| `--gpu` | 0 | GPU index |
| `--save-dir` | `''` | output folder |
| `--log-every` | 1 | print a log line every n steps |

A run is skipped if its result file already exists.

### Experiments of the paper

The scripts in `scripts/` reproduce the experiments of the paper:

| Script | Arguments | Runs |
| --- | --- | --- |
| `01_run__baseline.sh` | model | random blind search baseline, 20 runs |
| `02_run__greedy.sh` | model, permutations | greedy search, 20 runs |

The script `scripts/run_all.sh` runs all experiments for all the models used in
the paper.

```bash
uv run bash scripts/run_all.sh
```

The models are:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-Small-24B-Instruct-2501`

## Results

Each script writes JSON files under `results/`:

```
results/
├── baseline/model=<model>/prompt=<prompt>/n-permutations=1/seed=<s>.json
└── greedy/model=<model>/prompt=<prompt>/n-permutations=<K>/seed=<s>.json
```

A search result contains:

- **Search runs** contain `steps`, one entry per step with the loss, the
  words, the response and the answer probabilities, and `permutations`, the
  outcome for all 24 orderings of the final word set.

- `steps`: one entry per step, with the current words (query word last), the
  loss, the model's response, whether it is incorrect, and the probabilities of
  the incorrect and the correct answer.
    - `steps[0]` is the random initialization.
- `permutations`: the evaluation of the final word set over all 24 orderings
   with the outcome of each ordering.

## Analysis

`notebooks/explore.ipynb` loads all results and contains the analyses of the
paper.

## Repository structure

```
├── run__search.py      greedy search and random blind search baseline
├── resources/          prompt templates and task configs
├── scripts/            experiment scripts
├── notebooks/          analysis of the results
└── src/
    ├── model.py        model loading and chat template encoding
    ├── prompt.py       task configs, template filling, answer tokens
    ├── scoring.py      single forward pass: loss, response, success
    ├── proposals.py    random single-word replacements
    ├── permutation.py  sampling and evaluating set orderings
    ├── vocabulary.py   vocabulary construction
    ├── logs.py         step logs of a search run
    └── io.py           result paths
```
