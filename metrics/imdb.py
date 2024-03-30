import os, sys, json, csv
import evaluate
from pprint import pprint
from scorer import AccuracyScorer, NLIDiagScorer, HansScorer, NLIDiagScorerAccuracy
from metrics_utils import extract_model_details, write_list_of_dicts_as_csv
from copy import deepcopy
import pickle

FILE_NAME_TO_METRIC_DETAILS = {
        'robustness/twitter_emotion/twitter_emotion_2018.csv': {'evaluation': 'twitter emotion 2', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy', 'pred_label_map' :{'neg': 'negative', 'pos': 'positive',}},

        'robustness/semeval_2017_task_4_twitter/SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt': {'evaluation': 'twitter emotion', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy', 'pred_label_map' :{'neg': 'negative', 'pos': 'positive',}},

        'robustness/c-imdb/test_paired.tsv': {'evaluation': 'c-imdb', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy', 'pred_label_map' :{'neg': 'Negative', 'pos': 'Positive',}},

        'robustness/gardner_imdb/test_original.tsv': {'evaluation': 'imdb contrast', 'gold_key': 'Sentiment', 'type': 'none', 'subset': 'original', 'metric': 'accuracy', 'pred_label_map' :{'neg': 'Negative', 'pos': 'Positive'}},

        'robustness/gardner_imdb/test_contrast.tsv': {'evaluation': 'imdb contrast', 'gold_key': 'Sentiment', 'type': 'none', 'subset': 'contrast', 'metric': 'accuracy', 'pred_label_map' :{'neg': 'Negative', 'pos': 'Positive'}},

        'robustness/evaluation_sets/imdb/imdb_test.csv': {'evaluation': 'imdb', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy'},

        'robustness/evaluation_sets/imdb/rotten_tomatoes_validation.csv': {'evaluation': 'rotten_tomatoes', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy'},

        'robustness/evaluation_sets/imdb/sst2_validation.csv': {'evaluation': 'sst2', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy'},


        'robustness/evaluation_sets/imdb/yelp_polarity_test.csv': {'evaluation': 'yelp', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy'},

        'robustness/evaluation_sets/imdb/amazon_reviews_appliances_beauty_fashion_gift_cards_magazines_software_10000_evaluation.jsonl': {'evaluation': 'amazon_reviews', 'gold_key': 'label', 'type': 'none', 'subset': 'none', 'metric': 'accuracy', 'pred_label_map' :{'neg': 'negative', 'pos': 'positive'}}
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
