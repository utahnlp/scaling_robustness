import os, sys, json, csv
import evaluate
from pprint import pprint
from scorer import AccuracyScorer, NLIDiagScorer, HansScorer, NLIDiagScorerAccuracy
from metrics_utils import extract_model_details, write_list_of_dicts_as_csv
from copy import deepcopy
import pickle

FILE_NAME_TO_METRIC_DETAILS = {
        # Stress Test Evaluation Sets
        'robustness/stress_test_final/StressTests/Word_Overlap/multinli_0.9_taut2_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Word_Overlap', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Word_Overlap/multinli_0.9_taut2_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Word_Overlap', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_swap_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_swap', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_swap_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_swap', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_keyboard_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_keyboard', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_keyboard_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_keyboard', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_functionword_swap_perturbed', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_functionword_swap_perturbed_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_functionword_swap_perturbed', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_contentword_swap_perturbed', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Spelling_Error_contentword_swap_perturbed', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Numerical_Reasoning/multinli_0.9_quant_hard.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Numerical_Reasoning', 'subset': None, 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Negation/multinli_0.9_negation_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Negation', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Negation/multinli_0.9_negation_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Negation', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Length_Mismatch/multinli_0.9_length_mismatch_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Length_Mismatch', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Length_Mismatch/multinli_0.9_length_mismatch_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Length_Mismatch', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Antonym/multinli_0.9_antonym_mismatched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Antonym', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness/stress_test_final/StressTests/Antonym/multinli_0.9_antonym_matched.jsonl': {'evaluation': 'stress test', 'gold_key' :'gold_label', 'type': 'Antonym', 'subset': 'validation_matched', 'metric': 'accuracy'},

        # Breaking NLI
        'robustness/breaking_nli/data/dataset.csv': {'evaluation': 'breaking nli', 'gold_key' :'label', 'type': None, 'subset': None, 'metric': 'accuracy'},

        # SNLI Counterfactually Augmented Data (CAD)
        'robustness/snli_cad/dev.tsv': {'evaluation': 'snli cad', 'gold_key' :'gold_label', 'type': None, 'subset': 'dev', 'metric': 'accuracy'},
        'robustness/snli_cad/test.tsv': {'evaluation': 'snli cad', 'gold_key' :'gold_label', 'type': None, 'subset': 'test', 'metric': 'accuracy'},

        # NLI Diagnostics
        'robustness/nli-diagnostics/diagnostic-full.csv': {'evaluation': 'nli-diagnostics', 'gold_key' :'label', 'type': None, 'subset': None, 'metric': 'nli_diag,nli_diag_accuracy'},

        # Hans Heuristics
        'robustness/hans/hans/heuristics_evaluation_set.jsonl': {'evaluation': 'hans', 'gold_key' :'gold_label', 'type': None, 'subset': None, 'metric': 'hans', 'pred_label_map' :{'entailment': 'entailment', 'neutral': 'non-entailment', 'contradiction': 'non-entailment'}},

        # MNLI Validation/Hard
        'mnli': {'evaluation': 'mnli-matched', 'gold_key' :'label', 'type': None, 'subset': 'validation_matched', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness/evaluation_sets/mnli/mnli_validation_mismatched.csv': {'evaluation': 'mnli-mismatched', 'gold_key' :'label', 'type': None, 'subset': 'validation_mismatched', 'metric': 'accuracy'},

        'robustness/snli-mnli-hard/data/MNLIMatchedHardWithHardTest/test.tsv': {'evaluation': 'mnli-hard', 'gold_key' :'label', 'type': None, 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness/snli-mnli-hard/data/MNLIMismatchedHardWithHardTest/test.tsv': {'evaluation': 'mnli-hard', 'gold_key' :'label', 'type': None, 'subset': 'validation_mismatched', 'metric': 'accuracy'},

        # SNLI Validation/Hard
        'snli': {'evaluation': 'snli', 'gold_key' :'label', 'type': None, 'subset': 'validation', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness/snli-mnli-hard/data/SNLIHard/snli_1.0_test_hard.jsonl': {'evaluation': 'snli-hard', 'gold_key' :'gold_label', 'type': None, 'subset': 'validation', 'metric': 'accuracy'},

        # DNLI Test
        'robustness/dnli/dialogue_nli/dialogue_nli_verified_test_reformatted.jsonl': {'evaluation': 'dnli', 'gold_key' :'label', 'type': None, 'subset': 'validation', 'metric': 'accuracy', 'pred_label_map': {'entailment': 'positive', 'contradiction': 'negative', 'neutral': 'neutral'}},

        # ANLI Test
        'robustness/anli/test_r1.jsonl': {'evaluation': 'anli', 'gold_key' :'label', 'type': None, 'subset': 'test_r1', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness/anli/test_r2.jsonl': {'evaluation': 'anli', 'gold_key' :'label', 'type': None, 'subset': 'test_r2', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness/anli/test_r3.jsonl': {'evaluation': 'anli', 'gold_key' :'label', 'type': None, 'subset': 'test_r3', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        }

METRIC_TO_SCORER = {
        'accuracy': AccuracyScorer,
        'nli_diag': NLIDiagScorer,
        'nli_diag_accuracy': NLIDiagScorerAccuracy,
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

    #for f_name in os.listdir(model_dir):
    for eval_set_path, eval_set_details in FILE_NAME_TO_METRIC_DETAILS.items():
        f = eval_set_path.replace('/', '__') + '.preds.jsonl'
        #metric_details = FILE_NAME_TO_METRIC_DETAILS[f]
        if not os.path.isfile(model_dir + '/' + f):
            print('=='*10)
            print(f'Predictions for file {eval_set_path} not found. {f} does not exist.')
            print('=='*10)
            continue
        else:
            print('Processing ', model_dir + '/' + f)
        metric_details = eval_set_details
        eval_name = metric_details['evaluation']

        if 'pred_label_map' not in metric_details:
            pred_label_map = None
        else:
            pred_label_map = metric_details['pred_label_map']

        gold_key = metric_details['gold_key']

        eval_dataset = eval_name

        for metric_name in metric_details['metric'].split(','):
            if metric_name.strip() == '':
                continue
            metric = METRIC_TO_SCORER[metric_name](filename=model_dir + '/' + f, gold_key=gold_key, pred_key = 'predicted', pred_label_map=pred_label_map)
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
