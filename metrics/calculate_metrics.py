import os, sys, json, csv
from constants import INT_LABEL_TO_STR

def load_jsonl_file(filename):

    data = []
    with open(filename, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))

    return data

FILE_TO_LABEL_KEY_MAP = {
	'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Antonym/multinli_0.9_antonym_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Antonym/multinli_0.9_antonym_mismatched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Length_Mismatch/multinli_0.9_length_mismatch_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Length_Mismatch/multinli_0.9_length_mismatch_mismatched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Negation/multinli_0.9_negation_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Negation/multinli_0.9_negation_mismatched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Numerical_Reasoning/multinli_0.9_quant_hard.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_functionword_swap_perturbed_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_keyboard_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_keyboard_mismatched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_swap_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Spelling_Error/multinli_0.9_dev_gram_swap_mismatched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Word_Overlap/multinli_0.9_taut2_matched.jsonl': 'gold_label',
'robustness/stress_test_final/StressTests/Word_Overlap/multinli_0.9_taut2_mismatched.jsonl': 'gold_label',
'robustness/snli-mnli-hard/data/SNLIHard/snli_1.0_test_hard.jsonl': 'gold_label',
'robustness/snli-mnli-hard/data/MNLIMatchedHardWithHardTest/test.tsv': 'label',
'robustness/snli-mnli-hard/data/MNLIMismatchedHardWithHardTest/test.tsv': 'label',
'robustness/evaluation_sets/mnli/mnli_validation_mismatched.csv': 'label',
'robustness/hans/hans/heuristics_evaluation_set.jsonl': 'gold_label',
'robustness/nli-diagnostics/diagnostic-full.csv': 'label',
'robustness/snli_cad/dev.tsv': 'gold_label',
'robustness/snli_cad/test.tsv': 'gold_label',
'robustness/breaking_nli/data/dataset.csv': 'label',
        }

def calculate_accuracy(examples, task):
    LABEL_KEY = FILE_TO_LABEL_KEY_MAP.get(examples[0]['filename'], 'label')
    label_dict = INT_LABEL_TO_STR[task]
    #print(label_dict[2])
    correct = 0.0
    total = 0.0
    for ex in examples:
        if isinstance(ex[LABEL_KEY], int):
            if ex[LABEL_KEY] == -1:
                continue
            label = label_dict[ex[LABEL_KEY]]
        else:
            label = ex[LABEL_KEY]
            if label in ['not_entailement', 'not-entailment', 'non-entailment']:
                if ex['predicted'].lower().strip() != 'entailment':
                    correct +=1
            else:
                if label.lower().strip() == ex['predicted'].lower().strip():
                    correct +=1
        total +=1

    accuracy = correct*100 / total

    return accuracy


if __name__=="__main__":

    model_dir = sys.argv[1]
    task = sys.argv[2]

    evaluation_dir = model_dir + '/evaluations/'
    files = []
    for f in os.listdir(evaluation_dir):
        if 'preds.jsonl' in f:
            files.append(evaluation_dir + f)

    for f in files:
        #print(f)
        examples = load_jsonl_file(f)
        acc = calculate_accuracy(examples, task)
        print(f'File: {f}, Accuracy: {acc}')
