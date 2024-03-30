#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

import time
import torch
from constants import INT_LABEL_TO_STR_T5_TRAINING, INT_LABEL_TO_STR
from utils import write_as_jsonl_file
from utils import InputReader

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.24.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    #test_file: Optional[str] = field(
    #    default=None,
    #    metadata={
    #        "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
    #    },
    #)
    test_file: List[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    prediction_format: Optional[str] = field(default='default', metadata={"help": "What format to predict in"})

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #     raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "snli": ("premise", "hypothesis"),
    "pietrolesci/nli_fever": ('premise', 'hypothesis'),
    "boolq": ('question', 'passage'),
}

t5_task_prefix = {
    'sst2': 'Sentence: ',
    'rotten_tomatoes': 'Sentence: ',
    'ag_news': 'Sentence: ',
    'imdb': 'Sentence: ',
    'yelp_polarity': 'Sentence: ',
    'snli': ('snli premise: ', 'hypothesis: '),
    'mnli': ('mnli premise: ', 'hypothesis: '),
    'mrpc': ('mrpc sentence1: ', 'sentence2: '),
    'qqp': ('qqp question1: ', 'question2: '),
    'qnli': ('qnli question: ', 'Sentence: '),
    'pietrolesci/nli_fever': ('nli_fever premise', 'hypothesis'),
    "boolq": ('boolq question: ', 'passage: '),
}

GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'sst2', 'rte', 'wnli', 'stsb']

def transform_to_t5_targets(labels, task_name):
    targets = []

    #if task_name =='sst2':
    if task_name in ['sst2', 'imdb', 'yelp_polarity', 'rotten_tomatoes']:
        # sentiment with SST2 and other datasets, 0-> negative, 1 -> positive
        for lb in labels:
            if lb == 1:
                targets.append("positive")
            else:
                targets.append("negative")

    elif task_name in ['snli', 'mnli']:
        # both mnli and snli have same label map. {'entailment':0, 'neutral':1, 'contradiction':2}
        for lb in labels:
            if lb == 0:
                targets.append("entailment")
            elif lb == 1:
                targets.append("neutral")
            else:
                targets.append("contradiction")
    elif task_name in ['mrpc']:
        for lb in labels:
            if lb == 0:
                targets.append("not_equivalent")
            else:
                targets.append("equivalent")
    elif task_name in ['qqp']:
        for lb in labels:
            if lb == 0:
                targets.append("not_duplicate")
            else:
                targets.append("duplicate")
    elif task_name in ['ag_news']:
        for lb in labels:
            if lb ==0:
                targets.append("world")
            elif lb == 1:
                targets.append("sports")
            elif lb==2:
                targets.append("business")
            else:
                targets.append("sci/tech")
    elif task_name in ['qnli']:
        for lb in labels:
            if lb == 0:
                targets.append("entailment")
            else:
                targets.append("not_entailment")
    elif task_name in ['pietrolesci/nli_fever']:
        for lb in labels:
            if lb == 0:
                targets.append("entailment")
            elif lb == 1:
                targets.append("neutral")
            else:
                targets.append("contradiction")
    else:
        # do nothing
        return labels
    return targets

def transform_from_t5_targets(targets, task_name):
    labels = []

    if task_name in ['sst2', 'imdb', 'yelp_polarity', 'rotten_tomatoes']:
        for tar in targets:
            if tar.lower() == 'positive':
                labels.append(1)
            else:
                labels.append(0)
    elif task_name in ['mnli', 'snli']:
        for tar in targets:
            if tar.lower() == 'entailment':
                labels.append(0)
            elif tar.lower() == 'neutral':
                labels.append(1)
            else:
                labels.append(2)
    elif task_name in ['mrpc']:
        for lb in targets:
            if lb == "not_equivalent":
                labels.append(0)
            else:
                labels.append(1)
    elif task_name in ['qqp']:
        for lb in targets:
            if lb == "not_duplicate":
                labels.append(0)
            else:
                labels.append(1)
    elif task_name in ['ag_news']:
        for tar in targets:
            if tar.lower() == "world":
                labels.append(0)
            elif tar.lower() == "sports":
                labels.append(1)
            elif tar.lower() == 'business':
                labels.append(2)
            else:
                labels.append(3)
    elif task_name in ['qnli']:
        for lb in targets:
            if lb == "entailment":
                labels.append(0)
            else:
                labels.append(1)
    elif task_name in ['pietrolesci/nli_fever']:
        for tar in targets:
            if tar.lower() == 'entailment':
                labels.append(0)
            elif tar.lower() == 'neutral':
                labels.append(1)
            else:
                labels.append(2)
    else:
        # do nothing
        return targets
    return labels

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_prediction_prob(scores, task, tokenizer, scheme='first', return_prob=True):

    """
    Given logit scores for the generated sequence, return the probability of all the labels.
    There are two possible ways to calculate these scores.
    1. 'first': Using the logit score of only the first token of the class label
    2. 'average': Average the logit scores of all the tokens in the class label, normalized by the length

    Assume scores = LongTensor of shape (num_toks, vocab)

    In the 'first' setting, just find the token ids for the first decoded token of each of the labels and use that as the logit of the three labels.
    In the 'average' setting, for each of the labels, find the non-special token_ids (as a sequence) and take the scores for all of them and average them
    There is one problem in the 'average' scheme.
    If all the examples in a batch have the label 'duplicate', then the sequence length (num_toks) will never be long enough to calculate the logit score for 'not_duplicate'
    """

    label_to_tokenizer_ids = {}

    for lb_int, lb_str in INT_LABEL_TO_STR_T5_TRAINING[task.lower()].items():
        label_to_tokenizer_ids[lb_int] = list(tokenizer.encode(lb_str, add_special_tokens=False))

    """
    for example, we get {'not_duplicate': [59, 834, 26, 413, 26221], 'duplicate': [19197]}
    for the qqp task
    """

    if scheme == 'first':
        label_to_logits = [0] * len(label_to_tokenizer_ids)
        for key, val in label_to_tokenizer_ids.items():
            label_to_logits[key] = scores[0][val[0]]
        if return_prob:
            label_to_probs = torch.nn.functional.softmax(torch.Tensor(label_to_logits), dim=-1).tolist()
            label_to_logits = label_to_probs
        label_to_logits_dict = {}
        for i in range(len(label_to_logits)):
            label_to_logits_dict[INT_LABEL_TO_STR_T5_TRAINING[task.lower()][i]] = label_to_logits[i]
        return label_to_logits_dict


def do_logit_score_and_generation_sanity_check(logit_scores, generated_tok, tokenizer):

    logit_label = max(logit_scores, key=logit_scores.get)

    generated_label = tokenizer.decode(generated_tok, skip_special_tokens=True)

    print(logit_scores)
    print(generated_tok)
    print(logit_label, generated_label)
    assert logit_label.lower() == generated_label.lower()


def prediction_loop(model, trainer, tokenizer, predict_dataset, task):

    dataloader = trainer.get_test_dataloader(predict_dataset)
    start_time = time.time()

    trainer.model = trainer._wrap_model(trainer.model, training=False, dataloader=dataloader)


    batch_size = dataloader.batch_size
    num_examples = trainer.num_examples(dataloader)

    description = "Prediction "
    logger.info(f"***** Running {description} *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Batch size = {batch_size}")
    losses_host: torch.Tensor = None
    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
    labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
    inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None

    #world_size = max(1, args.world_size)
    world_size = 1

    trainer.model.eval()

    returned_outputs = []

    for step, inputs in enumerate(dataloader):
        inputs = trainer._prepare_inputs(inputs)
        gen_kwargs = {}
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = trainer.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else trainer.model.config.num_beams
        )
        default_synced_gpus = False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(trainer.model, "encoder") and trainer.model.encoder.main_input_name != trainer.model.main_input_name:
            generation_inputs = inputs[trainer.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[trainer.model.main_input_name]

        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True
        generated_dict = trainer.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        generated_tokens = generated_dict.sequences
        #print(generated_tokens)
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = trainer._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = trainer._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        generated_dict.sequences = generated_tokens


        for i in range(inputs['input_ids'].shape[0]):
            gen_toks = generated_dict.sequences.detach().cpu()[i].tolist()
            seq_scores = torch.zeros(len(generated_dict.scores), generated_dict.scores[0].shape[1]) # shape = (seq_len, vocab)
            for j in range(len(generated_dict.scores)):
                seq_scores[j] = generated_dict.scores[j][i]
            logit_scores = extract_prediction_prob(seq_scores, task, tokenizer)
            inps = inputs['input_ids'][0].tolist()
            #print(inps)
            #print(gen_toks)
            #print(logit_scores)
            #print(generated_dict.scores[0][i][19197], generated_dict.scores[0][i][59])
            #do_logit_score_and_generation_sanity_check(logit_scores, gen_toks, tokenizer)
            returned_outputs.append((inps, gen_toks, logit_scores))
    return returned_outputs

def extract_label_for_boolq(dataset):

    labels = []
    for ex in dataset:
        labels.append(str(ex['answer'])) # convert bool True to 'True'

    new_dataset = dataset.add_column("label", labels)

    return new_dataset

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.is_matched = False

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    data_args.source_prefix = t5_task_prefix[data_args.task_name]

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.task_name in GLUE_TASKS:
        raw_datasets = load_dataset("glue", data_args.task_name, revision='script')
    else:
        raw_datasets = load_dataset(data_args.task_name)

    if 'boolq' in data_args.task_name:
        new_raw_datasets = type(raw_datasets)()
        new_raw_datasets['train'] = extract_label_for_boolq(raw_datasets['train'])
        new_raw_datasets['validation'] = extract_label_for_boolq(raw_datasets['validation'])
        raw_datasets = new_raw_datasets
    #if data_args.task_name in ['snli', 'imdb', 'yelp_polarity']:
    #    raw_datasets = load_dataset(data_args.task_name)
    #else:
    #    raw_datasets = load_dataset("glue", data_args.task_name)

    if training_args.do_predict and data_args.test_file is not None:
        print("Loading the test files to make predictions, test files are : ", str(data_args.test_file))
            # To evaluate on a different set of examples
            #prediction_dataset = load_dataset("csv", data_files={'predict':data_args.test_file[0]})
            #print('Printing details of the loaded prediction file ')
            #print(prediction_dataset)
        input_readers = []
        for i, test_file in enumerate(data_args.test_file):
            if '^^' in test_file:
                # means the data has to be loaded from HF datasets
                data_name, split_name = test_file.strip().split('^^')
                input_reader = InputReader((data_name, split_name), split='predict_' + str(i), task_or_dataset=data_args.task_name if data_args.task_name is not None else data_args.dataset_name)
            else:
                input_reader = InputReader(test_file, split='predict_' + str(i), task_or_dataset=data_args.task_name if data_args.task_name is not None else data_args.dataset_name)
            input_readers.append(input_reader)
        prediction_dataset = input_reader.dataset
        print('Printing details of the loaded prediction file ')
        for input_reader in input_readers:
            print(input_reader.dataset)



    #label_list = raw_datasets["train"].features["label"].names
    #num_labels = len(label_list)


    if data_args.task_name in GLUE_TASKS:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        if data_args.task_name is not None and data_args.task_name in task_to_keys:
            sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print('=='*24)
    print('Number of trainable parameters = ', count_trainable_parameters(model))
    print('=='*24)

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    if isinstance(data_args.source_prefix, tuple):
        prefix = (data_args.source_prefix[0], data_args.source_prefix[1])
    else:
        prefix = (data_args.source_prefix, None)

    # prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # NOTE: Only works for SST2 for now # NOTE: Fixed I think now
    def preprocess_function(examples):

        # add prefixes for T5 classification

        if sentence2_key is None:
            ex_s1_key = [prefix[0] + inp for inp in examples[sentence1_key]]
            examples[sentence1_key] = ex_s1_key
        else:
            ex_s1_key = [prefix[0] + inp for inp in examples[sentence1_key]]
            examples[sentence1_key] = ex_s1_key
            ex_s2_key = [prefix[1] + inp for inp in examples[sentence2_key]]
            examples[sentence2_key] = ex_s2_key
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key],examples[sentence2_key])
        )
        # inputs = examples[sentence1_key]
        if "label" in examples and not training_args.do_predict:
            targets = transform_to_t5_targets(examples['label'], data_args.task_name)
        # targets = examples[summary_column]
        # inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(*inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        if "label" in examples and not training_args.do_predict:
            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 5):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            logger.info(f"As a string, this is ---- {tokenizer.decode(train_dataset[index]['input_ids'])}---- , with the targets as ----{tokenizer.decode(train_dataset[index]['labels'])}----")


    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        # if "validation" not in raw_datasets:
        #if "validation" not in raw_datasets and "validation_matched" not in raw_datasets and "validation_mismatched" not in raw_datasets:
        #    raise ValueError("--do_eval requires a validation dataset")
        #if data_args.is_matched and data_args.task_name == "mnli":
        #    val_subset = 'validation_matched'
        #else:
        #    val_subset = 'validation_mismatched'
        #if not data_args.task_name == "mnli":
        #    val_subset = 'validation'
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        #eval_dataset = raw_datasets[val_subset]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["validation_matched"].column_names if data_args.task_name == "mnli" else raw_datasets["validation"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.task_name == "mnli":
            eval_dataset_mm = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples_mm = min(len(eval_dataset_mm), data_args.max_eval_samples)
                eval_dataset_mm = eval_dataset_mm.select(range(max_eval_samples_mm))
            eval_dataset_mm = eval_dataset_mm.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["validation_mismatched"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation mismatched dataset",
            )

    if training_args.do_predict or data_args.test_file is not None:
        max_target_length = data_args.val_max_target_length
        if data_args.test_file is None:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            #with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            #    predict_dataset = prediction_dataset.map(
            #        preprocess_function,
            #        batched=True,
            #        num_proc=data_args.preprocessing_num_workers,
            #        load_from_cache_file=False,
            #        desc="Running tokenizer on prediction dataset",
            #    )
        else:
            # use the data_args.test_file
            predict_datasets_list = []
            for ir in input_readers:
                prediction_dataset = ir.dataset
                predict_dataset = prediction_dataset.map(
                                preprocess_function,
                                batched=True,
                                load_from_cache_file=False,
                                desc="Running tokenizer on dataset",
                )
                predict_datasets_list.append(predict_dataset)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )


    if data_args.task_name == 'snli':
        metric = evaluate.load("glue", "mnli") # evaluation metric same as mnli
    elif data_args.task_name in GLUE_TASKS:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    def postprocess_text(preds, labels):

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # # rougeLSum expects newline after each sentence
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return transform_from_t5_targets(preds, data_args.task_name), transform_from_t5_targets(labels, data_args.task_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        num_correct = sum([1 for i in range(len(decoded_preds)) if decoded_preds[i] == decoded_labels[i]])
        total = len(decoded_preds)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        result['num_correct'] = num_correct
        result['total'] = total
        result['my_acc'] = float(num_correct)/total
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        combined = {}
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = eval_dataset_mm
            eval_datasets.append(valid_mm_dataset)
        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset, max_length=max_length, num_beams=num_beams)
            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            #max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            print(task, len(eval_dataset))
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)
            #trainer.save_metrics("eval", metrics)
            print(combined)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        task = data_args.task_name if data_args.task_name is not None else data_args.dataset_name
        for dataset_itr in range(len(input_readers)):
            input_reader = input_readers[dataset_itr]
            predict_dataset = predict_datasets_list[dataset_itr][input_reader.split]
            if 'label' in predict_dataset.column_names:
                predict_dataset = predict_dataset.remove_columns("label")
            returned_outputs = prediction_loop(model, trainer, tokenizer, predict_dataset, task)
            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
        #    #predict_results = trainer.predict(
        #    #    predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        #    #)
        #if data_args.test_file is not None:
        #    predict_datasets = [predict_dataset['predict']]
        #else:
        #    predict_datasets = [predict_dataset]

        #for predict_dataset in predict_datasets:
        #    if 'label' in predict_dataset.column_names:
        #        predict_dataset = predict_dataset.remove_columns("label")
        #    returned_outputs = prediction_loop(model, trainer, tokenizer, predict_dataset, data_args.task_name)
        #    #predict_results = trainer.predict(
        #    #    predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        #    #)
        #    #metrics = predict_results.metrics
        #    #max_predict_samples = (
        #    #    data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        #    #)
        #    #metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        #    #trainer.log_metrics("predict", metrics)
        #    #trainer.save_metrics("predict", metrics)
            #test_file_name = data_args.test_file.split('/')[-1].split('.')[0].strip()
            #output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.task_name}_{test_file_name}.txt")
            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    if data_args.prediction_format.lower() == 'orig_default':
        #                # NOT WORKING YET
                        predictions = tokenizer.batch_decode(
                            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        predictions = [pred.strip() for pred in predictions]
                        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                        with open(output_prediction_file, "w") as writer:
                            writer.write("\n".join(predictions))
                    elif data_args.prediction_format.lower() == 'checklist_qqp':
                        LABEL_TO_PRINT = 'duplicate'
                        with open(output_predict_file, "w") as writer:
                            logger.info(f"***** Predict results {data_args.task_name} *****")
                            for index in range(len(returned_outputs)):
                                pred_prob = returned_outputs[index][-1][LABEL_TO_PRINT]
                                writer.write(str(pred_prob) + '\n')

                    elif data_args.prediction_format.lower() == 'default':
                        name = data_args.task_name
                        label_id_to_str = INT_LABEL_TO_STR[name]
                        #print(len(prediction_dataset['predict']))
                        #print(len(returned_outputs))
                        #print(returned_outputs[0])
                        #prediction_dataset = prediction_dataset['predict']
                        #print(len(prediction_dataset), prediction_probs.shape[0])
                        prediction_dataset = input_reader.dataset[input_reader.split]
                        assert len(prediction_dataset) == len(returned_outputs)
                        prediction_dataset = input_reader.dataset[input_reader.split]
                        output_data = []
                        for ex_idx in range(len(prediction_dataset)):
                            ex = {}
                            ex['filename'] = input_reader.file_path_or_name
                            for k, v in prediction_dataset[ex_idx].items():
                                ex[k] = v
                            predicted_label = max(returned_outputs[ex_idx][-1], key=returned_outputs[ex_idx][-1].get)
                            probs_dict = returned_outputs[ex_idx][-1]
                            ex['probs'] = probs_dict
                            ex['predicted'] = predicted_label
                            output_data.append(ex)
                        input_reader.write_predictions(output_data, training_args.output_dir, input_reader.file_path_or_name)
                        #output_predict_file = os.path.join(training_args.output_dir, data_args.test_file.split('/')[-1] + ".preds.jsonl")
                        #write_as_jsonl_file(output_predict_file, output_data)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    data_args.dataset_name = data_args.task_name
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
