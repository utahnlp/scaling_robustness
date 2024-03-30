import os, sys, json, csv
import evaluate
from pprint import pprint
from scorer import AccuracyScorer, NLIDiagScorer, HansScorer, F1Scorer
from metrics_utils import extract_model_details, write_list_of_dicts_as_csv
from copy import deepcopy
import pickle

FILE_NAME_TO_METRIC_DETAILS = {
        'robustness/twitter_ppdb/twitter_url_corpus_test.csv.preds.jsonl': {'evaluation': 'twitter_ppdb', 'gold_key': 'label', 'type': None, 'subset': None, 'metric': 'accuracy'},

        'robustness/evaluation_sets/qqp/paws_qqp_dev_test.csv.preds.jsonl': {'evaluation': 'paws_qqp', 'gold_key': 'label', 'type': None, 'subset': None, 'metric': 'accuracy'},

        'robustness/evaluation_sets/qqp/qqp_validation.csv.preds.jsonl': {'evaluation': 'qqp_validation', 'gold_key': 'label', 'type': None, 'subset': None, 'metric': 'accuracy'},

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
        f1 = F1Scorer(filename=model_dir + '/' + f, gold_key=gold_key, pred_key = 'predicted', pred_label_map={'not_duplicate': 0, 'duplicate': 1}, gold_label_map={'not_duplicate': 0, 'duplicate': 1})
        f1.calculate()

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
            row['f1'] = f1.metric
            list_of_rows.append(row)

    write_list_of_dicts_as_csv(output_file, list_of_rows, file_flag='a')
