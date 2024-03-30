import os, sys, json, csv
import evaluate
from pprint import pprint
from scorer import AccuracyScorer, NLIDiagScorer, HansScorer
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
        'robustness__stress_test_final__StressTests__Numerical_Reasoning__multinli_0.9_quant_hard.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Numerical_Reasoning', 'subset': 'none', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Negation__multinli_0.9_negation_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Negation', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Negation__multinli_0.9_negation_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Negation', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Length_Mismatch__multinli_0.9_length_mismatch_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Length_Mismatch', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Length_Mismatch__multinli_0.9_length_mismatch_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Length_Mismatch', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Antonym__multinli_0.9_antonym_mismatched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Antonym', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__stress_test_final__StressTests__Antonym__multinli_0.9_antonym_matched.jsonl.preds.jsonl': {'evaluation': 'stress_test', 'gold_key' :'gold_label', 'type': 'Antonym', 'subset': 'validation_matched', 'metric': 'accuracy'},

        # Breaking NLI
        'robustness__breaking_nli__data__dataset.csv.preds.jsonl': {'evaluation': 'breaking_nli', 'gold_key' :'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy'},

        # SNLI Counterfactually Augmented Data (CAD)
        'robustness__snli_cad__dev.tsv.preds.jsonl': {'evaluation': 'snli_cad', 'gold_key' :'gold_label', 'type': 'none', 'subset': 'dev', 'metric': 'accuracy'},
        'robustness__snli_cad__test.tsv.preds.jsonl': {'evaluation': 'snli_cad', 'gold_key' :'gold_label', 'type': 'none', 'subset': 'test', 'metric': 'accuracy'},

        # NLI Diagnostics
        'robustness__nli-diagnostics__diagnostic-full.csv.preds.jsonl': {'evaluation': 'nli_diagnostics', 'gold_key' :'label', 'type': 'none', 'subset': 'none', 'metric': 'nli_diag'},

        # Hans Heuristics
        'robustness__hans__hans__heuristics_evaluation_set.jsonl.preds.jsonl': {'evaluation': 'hans', 'gold_key' :'gold_label', 'type': 'none', 'subset': 'none', 'metric': 'hans', 'pred_label_map' :{'entailment': 'entailment', 'neutral': 'non-entailment', 'contradiction': 'non-entailment'}},

        # MNLI Validation/Hard
        'mnli.preds.jsonl': {'evaluation': 'mnli_validation', 'gold_key' :'label', 'type': 'none', 'subset': 'validation_matched', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness__evaluation_sets__mnli__mnli_validation_mismatched.csv.preds.jsonl': {'evaluation': 'mnli_validation', 'gold_key' :'label', 'type': 'none', 'subset': 'validation_mismatched', 'metric': 'accuracy'},
        'robustness__snli-mnli-hard__data__MNLIMatchedHardWithHardTest__test.tsv.preds.jsonl': {'evaluation': 'mnli_hard', 'gold_key' :'label', 'type': 'none', 'subset': 'validation_matched', 'metric': 'accuracy'},
        'robustness__snli-mnli-hard__data__MNLIMismatchedHardWithHardTest__test.tsv.preds.jsonl': {'evaluation': 'mnli_hard', 'gold_key' :'label', 'type': 'none', 'subset': 'validation_mismatched', 'metric': 'accuracy'},

        # SNLI Validation/Hard
        'snli.preds.jsonl': {'evaluation': 'snli_validation', 'gold_key' :'label', 'type': 'none', 'subset': 'validation', 'metric': 'accuracy', 'pred_label_map' : {'entailment': 0, 'neutral': 1, 'contradiction':2}},
        'robustness__snli-mnli-hard__data__SNLIHard__snli_1.0_test_hard.jsonl.preds.jsonl': {'evaluation': 'snli_hard', 'gold_key' :'gold_label', 'type': 'none', 'subset': 'validation', 'metric': 'accuracy'},

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
    #output_file = sys.argv[2]
    model_name = sys.argv[2]
    seed = sys.argv[3]
    output_file = 'mnli_pkls/' + model_name.replace('/', '#') + '_' + seed

    # we will report numbers in three ways:
    # 1. For each subset, report all the numbers we can (for example for stress tests, each type has its own accuracy)
    # 2. For each subset, we report one number (which is averaged over all subsets/types)
    # 3. A summary number, that is averaged over all subsets (for this we need the metric to be same for all of them - choose accuracy)

    # we wi

    eval_to_metrics = {}
    accuracy_eval_to_metrics = {}

    avg_eval_to_metrics = {}

    for f in os.listdir(model_dir):
        print(f)
        if not f.endswith('.preds.jsonl'):
            continue
        if f not in FILE_NAME_TO_METRIC_DETAILS:
            raise ValueError('Could not find details for file ', f)
        metric_details = FILE_NAME_TO_METRIC_DETAILS[f]
        eval_name = metric_details['evaluation']
        if eval_name not in eval_to_metrics:
            eval_to_metrics[eval_name] = {}
            avg_eval_to_metrics[eval_name] = {}
            accuracy_eval_to_metrics[eval_name] = {}
        key = (metric_details['type'], metric_details['subset'])
        if 'pred_label_map' not in metric_details:
            pred_label_map = None
        else:
            pred_label_map = metric_details['pred_label_map']
        gold_key = metric_details['gold_key']
        metric = METRIC_TO_SCORER[metric_details['metric']](filename=model_dir + '/' + f, gold_key=gold_key, pred_key = 'predicted', pred_label_map=pred_label_map)
        metric.calculate()
        eval_to_metrics[eval_name][key] = (metric.metric, metric.metric_name)
        accuracy_eval_to_metrics[eval_name][key] = (metric.metric, 'accuracy')
        avg_eval_to_metrics[eval_name][key] = (calculate_average(metric.metric), metric.metric_name)
        if eval_name == 'nli_diagnostics':
            pprint(metric.metric_accuracy)

    pprint(eval_to_metrics)
    pprint(avg_eval_to_metrics)
    ss

    data = {}
    #data['model_dir'] = model_dir
    #model_name =  model_dir.split('/')[-4] #'final_models/mnli/t5-11b/seed-1/evaluations/'
    #seed = int(model_dir.split('/')[-3].split('-')[1])
    #dataset = model_dir.split('final_models/')[-1].split('/')[0]
    data['model'] = model_name.replace('/', '#')
    data['seed'] = seed
    data['dataset'] = 'mnli'
    data['metrics'] = eval_to_metrics
    data['avg_metrics'] = avg_eval_to_metrics
    data['accuracy_metrics'] = accuracy_eval_to_metrics

    save_as_pkl_file(output_file, data)
