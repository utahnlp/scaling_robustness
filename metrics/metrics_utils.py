import os, sys, json, csv
import evaluate
from pprint import pprint

sys.path.append("../")
from utils import extract_model_details_from_path

def extract_model_details(model_path):

    dataset, model_name, model_name_key, params, seed = extract_model_details_from_path(model_path)

    model_details = {'train_dataset': dataset, 'model_name': model_name_key,
            'model_full_name': model_name, 'params': params, 'seed': seed}

    return model_details

def write_list_of_dicts_as_csv(filename , list_of_dicts, file_flag='w'):

    keys = list_of_dicts[0].keys()

    with open(filename, file_flag, encoding='utf8', newline='') as fp:
        dict_writer = csv.DictWriter(fp, keys)
        #dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)


if __name__=="__main__":

    model_path = '/uufs/chpc.utah.edu/common/home/u1266434/scr/robustness/t5_robustness/final_models/mnli/facebook/opt-13b/seed-1'
    format_results(None, model_path, {})
    model_path = '/uufs/chpc.utah.edu/common/home/u1266434/scr/robustness/t5_robustness/final_models/mnli/facebook/opt-13b/seed-1/'
    format_results(None, model_path, {})

    model_path = '/uufs/chpc.utah.edu/common/home/u1266434/scr/robustness/t5_robustness/final_models/mnli/t5-11b/seed-1/'
    format_results(None, model_path, {})
