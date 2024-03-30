import os, sys, json, csv
import evaluate
from pprint import pprint
from scorer import AccuracyScorer, NLIDiagScorer, HansScorer
from metrics_utils import extract_model_details, write_list_of_dicts_as_csv
from copy import deepcopy
import pickle

FILE_NAME_TO_METRIC_DETAILS = {
        # Stress Test Evaluation Sets
        'robustness__stress_test_final__StressTests__Word_Overlap__multinli_0.9_taut2_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Word_Overlap', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Word_Overlap__multinli_0.9_taut2_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Word_Overlap', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_swap_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_swap', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_swap_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_swap', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_keyboard_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_keyboard', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_keyboard_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_keyboard', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_functionword_swap_perturbed', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_functionword_swap_perturbed_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_functionword_swap_perturbed', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_contentword_swap_perturbed', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Spelling_Error__multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_contentword_swap_perturbed', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Numerical_Reasoning__multinli_0.9_quant_hard.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Numerical_Reasoning', 'subset': None, 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Negation__multinli_0.9_negation_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Negation', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Negation__multinli_0.9_negation_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Negation', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Length_Mismatch__multinli_0.9_length_mismatch_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Length_Mismatch', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Length_Mismatch__multinli_0.9_length_mismatch_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Length_Mismatch', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Antonym__multinli_0.9_antonym_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Antonym', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Antonym__multinli_0.9_antonym_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Antonym', 'subset': 'validation_matched', 'metric': 'accuracy'},

        # Breaking NLI
        'robustness__breaking_nli__data__dataset.csv.preds.jsonl': {'evaluation': 'breaking_nli', 'gold_key' :'label', 'type': None, 'subset': None, 'metric': 'accuracy'},

        # SNLI Counterfactually Augmented Data (CAD)
        'robustness__snli_cad__dev.tsv.preds.jsonl': {'evaluation': 'snli_cad', 'gold_key' :'gold_label', 'type': None, 'subset': 'dev', 'metric': 'accuracy'},
        'robustness__snli_cad__test.tsv.preds.jsonl': {'evaluation': 'snli_cad', 'gold_key' :'gold_label', 'type': None, 'subset': 'test', 'metric': 'accuracy'},

        # NLI Diagnostics
        'robustness__nli-diagnostics__diagnostic-full.csv.preds.jsonl': {'evaluation': 'nli_diagnostics', 'gold_key' :'label', 'type': None, 'subset': None, 'metric': 'nli_diag'},

        # Hans Heuristics
        'robustness__hans__hans__heuristics_evaluation_set.jsonl.preds.jsonl': {'evaluation': 'hans', 'gold_key' :'gold_label', 'type': None, 'subset': None, 'metric': 'hans', 'pred_label_map' :{'entailment': 'entailment', 'neutral': 'non-entailment', 'contradiction': 'non-entailment'}},

        # MNLI Validation/Hard
        'mnli.preds.jsonl': {'evaluation': 'mnli_validation', 'gold_key' :'label', 'type': None, 'subset': 'validation_matched', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness__evaluation_sets__mnli__mnli_validation_mismatched.csv.preds.jsonl': {'evaluation': 'mnli_validation', 'gold_key' :'label', 'type': None, 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__snli-mnli-hard__data__MNLIMatchedHardWithHardTest__test.tsv.preds.jsonl': {'evaluation': 'mnli_hard', 'gold_key' :'label', 'type': None, 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__snli-mnli-hard__data__MNLIMismatchedHardWithHardTest__test.tsv.preds.jsonl': {'evaluation': 'mnli_hard', 'gold_key' :'label', 'type': None, 'subset': 'validation_mismatched', 'metric': 'accuracy'},

        # SNLI Validation/Hard
        'snli.preds.jsonl': {'evaluation': 'snli_validation', 'gold_key' :'label', 'type': None, 'subset': 'validation', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness__snli-mnli-hard__data__SNLIHard__snli_1.0_test_hard.jsonl.preds.jsonl': {'evaluation': 'snli_hard', 'gold_key' :'gold_label', 'type': None, 'subset': 'validation', 'metric': 'accuracy'},

        # DNLI Test
        'robustness__dnli__dialogue_nli__dialogue_nli_verified_test_reformatted.jsonl.preds.jsonl': {'evaluation': 'dnli', 'gold_key' :'label', 'type': None, 'subset': 'validation', 'metric': 'accuracy', 'pred_label_map': {'entailment': 'positive', 'contradiction': 'negative', 'neutral': 'neutral'}},
        }

METRIC_TO_SCORER = {
        'accuracy': AccuracyScorer,
        'nli_diag': NLIDiagScorer,
        'hans': HansScorer,
        }


def calculate_average(d):
    if isinstance(d, dict):
        sum(d.values())/len(d)
    else:
        return d

def save_as_pkl_file(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

if __name__=="__main__":

    model_dir = sys.argv[1]
    output_file = sys.argv[2]

    # we will have a csv file with the following columns (among other details):
    # test_set: evaluation set name (ex: stress test)
    # category: say, Antonym
    # sub category: sub category of antonym (validation_matched or validation_mismatched)

    eval_to_metrics = {}
    accuracy_eval_to_metrics = {}

    avg_eval_to_metrics = {}

    model_details = extract_model_details(model_dir.strip('evaluations/'))
    list_of_rows = []

    for f in os.listdir(model_dir):
        print(f)
        if not f.endswith('.preds.jsonl'):
            continue
        if f not in FILE_NAME_TO_METRIC_DETAILS:
            raise ValueError('Could not find details for file ', f)
        metric_details = FILE_NAME_TO_METRIC_DETAILS[f]
        eval_name = metric_details['evaluation']

        if 'pred_label_map' not in metric_details:
            pred_label_map = None
        else:
            pred_label_map = metric_details['pred_label_map']

        gold_key = metric_details['gold_key']

        eval_dataset = eval_name

        metric = METRIC_TO_SCORER[metric_details['metric']](filename=model_dir + '/' + f, gold_key=gold_key, pred_key = 'predicted', pred_label_map=pred_label_map)
        metric.calculate()

        category = metric_details['type']
        if category is None:
            category = ''
            category_str = category
        else:
            category_str = category + ' | '

        sub_category = metric_details['subset']
        if sub_category is None:
            sub_category = ''
            sub_category_str = sub_category
        else:
            sub_category_str = sub_category + ' | '

        if isinstance(metric.metric, dict):
            for met_key, met_val in metric.metric.items():
                cat = category_str + sub_category_str + met_key[0]
                sub_cat = met_key[1]
                row = deepcopy(model_details)
                row['eval_dataset'] = eval_dataset
                row['category'] = cat
                row['sub_category'] = sub_cat
                row['score'] = met_val
                row['metric'] = metric.metric_name
                list_of_rows.append(row)
        else:
            metric_number = metric.metric
            cat = category
            sub_cat = sub_category
            row = deepcopy(model_details)
            row['eval_dataset'] = eval_dataset
            row['category'] = cat
            row['sub_category'] = sub_cat
            row['score'] = metric_number
            row['metric'] = metric.metric_name
            list_of_rows.append(row)

    write_list_of_dicts_as_csv(output_file, list_of_rows, file_flag='a')
