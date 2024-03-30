# Parameter Counts of pre-trained models

PARAMETER_COUNTS = {
        'roberta-base': 124647939,
        'roberta-large':355361794,
        'deberta-v3-base': 184423682,
        'deberta-v3-large': 435063810,
        'gpt2': 124442112,
        'gpt2-medium': 354826240,
        'gpt2-large': 774033920,
        'gpt2-xl': 1557616000,
        'opt-125m': 125240832,
        'opt-350m': 331197440,
        'opt-1.3b': 1315762176,
        'opt-2.7b': 2651601920,
        'opt-6.7b': 6658482176,
        'opt-13b': 12853483520,
        'tulu-2-dpo-13b': 12853483520,
        'tulu-2-13b-0-shot': 12853483520,
        'mistral-7b-0-shot': 6658482176,
        'mistral-7b-8-shot': 6658482176,
        'tulu-2-13b-8-shot': 12853483520,
        'llama2-13b-hf': 12853483520,
        't5-small': 60506624,
        't5-base': 222903552,
        't5-large': 737668096,
        't5-3b': 2851598336,
        't5-11b': 11307321344,
        }

TASKS_TO_COLUMN_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "pietrolesci/nli_fever": ('premise', 'hypothesis'),
    "ag_news": ("text", None),
    "yelp_polarity": ("text", None),
    "imdb": ("text", None),
    "rotten_tomatoes": ("text", None),
}

#
COLUMN_MAPPING = {
        'text': 'sentence',
        'sentence': 'text'
        }

LABEL_MAPPING = {
        ('imdb', 'sst2'): {'neg': 'negative', 'pos': 'positive',},
        ('sst2', 'imdb'): {'positive': 'pos', 'negative': 'neg',},

        ('rotten_tomatoes', 'sst2'): {'neg': 'negative', 'pos': 'positive',},
        ('sst2', 'rotten_tomatoes'): {'positive': 'pos', 'negative': 'neg',},

        ('yelp_polarity', 'sst2'): {'1': 'negative', '2': 'positive'},
        ('sst2', 'yelp_polarity'): {'positive': '2', 'negative': '1'},

        ('yelp_polarity', 'imdb'): {'1': 'neg', '2': 'pos'},
        ('imdb', 'yelp_polarity'): {'pos': '2', 'neg': '1'},

        ('yelp_polarity', 'rotten_tomatoes'): {'1': 'neg', '2': 'pos'},
        ('rotten_tomatoes', 'yelp_polarity'): {'pos': '2', 'neg': '1'},
        }


INT_LABEL_TO_STR = {
        'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'snli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'qnli': {0: 'entailment', 1: 'not_entailment'},
        'ag_news': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'},
        'qqp': {0: 'not_duplicate', 1: 'duplicate'},
        'mrpc': {0: 'not_equivalent', 1: 'equivalent'},
        'sst2': {0: 'negative', 1: 'positive'},
        'rotten_tomatoes': {0: 'neg', 1: 'pos'},
        'imdb': {0: 'neg', 1: 'pos'},
        'yelp_polarity': {0: '1', 1: '2'},
        'boolq': {0: 'False', 1: 'True'}, # {'False': 0, 'True': 1}
        "pietrolesci/nli_fever":{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        }

INT_LABEL_TO_STR_T5_TRAINING = {
        'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'snli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'qnli': {0: 'entailment', 1: 'not_entailment'},
        'ag_news': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'},
        'qqp': {0: 'not_duplicate', 1: 'duplicate'},
        'mrpc': {0: 'not_equivalent', 1: 'equivalent'},
        'sst2': {0: 'negative', 1: 'positive'},
        'imdb': {0: 'negative', 1: 'positive'},
        'rotten_tomatoes': {0: 'negative', 1: 'positive'},
        'yelp_polarity': {0: 'negative', 1: 'positive'},
        "pietrolesci/nli_fever":{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        }
