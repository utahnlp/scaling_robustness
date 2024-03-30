import os, sys, json, csv
import pickle
from pprint import pprint

def load_pkl_file(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data

def write_as_csv_file(filename, header, rows):

    fp = open(filename, 'w')
    csvwriter = csv.writer(fp, lineterminator='\n')

    csvwriter.writerow(header)

    for row in rows:
        csvwriter.writerow(row)

    fp.close()


if __name__=="__main__":

    all_metrics_dir = sys.argv[1]

    all_pkl_files = [all_metrics_dir + '/' + f for f in os.listdir(all_metrics_dir) if not f.endswith('.pkl')]
    print(all_pkl_files)

    model_to_metrics_dict = {}
    model_to_metrics_dict_detailed = {}

    for pkl in all_pkl_files:
        print('Pickle file ', pkl)
        data = load_pkl_file(pkl)
        model = data['model']
        seed = data['seed']
        dataset = data['dataset']
        metrics = data['metrics']
        dict_key = (dataset, model)
        if dict_key not in model_to_metrics_dict:
            model_to_metrics_dict[dict_key] = {}
            model_to_metrics_dict_detailed[dict_key] = {}
        for eval_set, met in metrics.items():
            if eval_set not in model_to_metrics_dict[dict_key]:
                model_to_metrics_dict[dict_key][eval_set] = {'metric_name': None, 'number': None}
                model_to_metrics_dict_detailed[dict_key][eval_set] = {}
            list_of_numbers = []
            dict_of_items = {}
            for key, val in met.items():
                model_to_metrics_dict[dict_key][eval_set]['metric_name'] = val[1]
                if isinstance(val[0], dict):
                    list_of_numbers.extend(list(val[0].values()))
                else:
                    list_of_numbers.append(val[0])
            for key, val in met.items():
                if str(key[0]) == 'none':
                    key0_str = ''
                else:
                    key0_str = str(key[0])
                if str(key[1]) == 'none':
                    key1_str = ''
                else:
                    key1_str = str(key[1])
                if isinstance(val[0], dict):
                    for k, v in val[0].items():
                        extra_key = str(k)
                        dict_of_items[key0_str + '|' + key1_str + '|' + extra_key] = v
                else:
                    extra_key = ''
                    dict_of_items[key0_str + '|' + key1_str ] = val
            model_to_metrics_dict_detailed[dict_key][eval_set] = dict_of_items


            average = sum(list_of_numbers) / len(list_of_numbers)
            model_to_metrics_dict[dict_key][eval_set]['number'] = average

    #pprint(model_to_metrics_dict)
    pprint(model_to_metrics_dict_detailed)
    model_to_metrics_dict_detailed_new = {}
    for key, val in model_to_metrics_dict_detailed.items():
        new_key = key[0] + '#' + key[1]
        model_to_metrics_dict_detailed_new[new_key] = val

    simplified_csv = []
    header = ['dataset', 'model', 'breaking_nli', 'hans', 'mnli_hard', 'mnli_validation', 'nli_diagnostics', 'snli_cad', 'snli_hard', 'snli_validation', 'stress_test']
    rows = []
    for key, val in model_to_metrics_dict.items():
        row = []
        row.append(key[0])
        row.append(key[1])
        for subset in header[2:]:
            if subset in val:
                row.append(val[subset]['number'])
            else:
                row.append(-1)
        rows.append(row)
    #write_as_csv_file('averaged_mnli.csv', header, rows)

    with open('mnli_results.json', 'w') as fp:
        json.dump(model_to_metrics_dict_detailed_new, fp)
