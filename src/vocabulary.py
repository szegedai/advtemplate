###=============================================================================
### Imports
###=============================================================================

import spacy
from better_profanity import profanity
from wordfreq import top_n_list, zipf_frequency

###=============================================================================
### Functionality
###=============================================================================

def build_vocabulary(
    n_words=50000,
    min_frequency=3.2,
    min_chars=5,
    language='en'
):
    # load top words
    words = top_n_list(language, n_words)

    # lemmatize words
    words = lemmatize_words(words)

    # keep alphanumerical words
    words = filter_alphabetic(words)

    # filter for minimum word frequency
    words = filter_frequency(words, min_frequency, language)

    # filter for minimum word length
    words = filter_min_chars(words, min_chars)

    # filter profanities
    words = filter_profanities(words)

    # deduplicate words
    words = sorted(set(words))

    return words

###-----------------------------------------------------------------------------
### Helpers
###-----------------------------------------------------------------------------

def filter_alphabetic(words):
    return [w for w in words if w.isalpha()]


def filter_frequency(words, min_frequency, language):
    return [w for w in words if zipf_frequency(w, language) >= min_frequency]


def filter_min_chars(words, min_chars):
    return [w for w in words if len(w) >= min_chars]


def filter_profanities(words):
    return [w for w in words if not profanity.contains_profanity(w)]


def lemmatize_words(words):
    # load model for morphological analysis
    nlp = spacy.load('en_core_web_sm')

    # lemmatize words
    lemmas = list()
    for doc in nlp.pipe(words, batch_size=256):
        # take lowercase base form of word
        lemma = doc[0].lemma_.lower()

        # skip empty lemma
        if lemma:
            lemmas.append(lemma)

    return lemmas
