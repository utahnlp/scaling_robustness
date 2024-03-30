import os, sys, csv
import json
from datasets import load_dataset
import datasets
import pandas as pd
from constants import PARAMETER_COUNTS

csv.field_size_limit(sys.maxsize)

def extract_model_details_from_path(model_path):
    model_path = model_path.split('final_models/')[1]
    model_path = model_path.replace('//', '/').strip()

    if 'pietrolesci' in model_path:
        split = model_path.split('/')
        dataset = split[0] + '/' + split[1]
        model_path = '/'.join(split[2:])
    elif 'alisawuffles' in model_path:
        split = model_path.split('/')
        dataset = split[0] + '/' + split[1]
        model_path = '/'.join(split[2:])
    else:
        split = model_path.split('/')
        dataset = model_path.split('/')[0]
        model_path = '/'.join(split[1:])

    if 'facebook' in model_path or 'microsoft' in model_path:
        split = model_path.split('/')
        model_name = split[0] + '/' + split[1]
        model_path = '/'.join(split[2:])
        params = PARAMETER_COUNTS[split[1]]
        model_name_key = split[1]
    else:
        split = model_path.split('/')
        model_name = split[0]
        model_path = '/'.join(split[1:])
        params = PARAMETER_COUNTS[split[0]]
        model_name_key = split[0]

    seed = int(model_path.split('/')[0].strip('seed-'))

    return dataset, model_name, model_name_key, params, seed


def write_as_csv_file(filename, header, rows):

    fp = open(filename, 'w')
    csvwriter = csv.writer(fp, lineterminator='\n')

    csvwriter.writerow(header)

    for row in rows:
        csvwriter.writerow(row)

    fp.close()

def load_csv_file(filename):

    header = 0
    rows = []
    with open(filename, 'r') as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            rows.append(row)

    return header, rows

def load_tsv_file(filename):

    header = 0
    rows = []
    with open(filename, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            rows.append(row)

    return header, rows

def load_csv_file_as_dict(filename):

    rows = []
    with open(filename, 'r') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)

    return rows

def load_tsv_file_as_dict(filename):

    rows = []
    with open(filename, 'r') as fp:
        reader = csv.DictReader(fp, delimiter='\t')
        for row in reader:
            rows.append(row)

    return rows

def load_jsonl_file(filename):

    data = []
    with open(filename, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))


    return data

def load_json_file(filename):

    with open(filename, 'r') as fp:
        data = json.load(fp)

    return data

def write_as_jsonl_file(filename, list_of_dicts):

    json_lines = [json.dumps(d) for d in list_of_dicts]
    json_data = '\n'.join(json_lines)

    with open(filename, 'w') as fp:
        fp.write(json_data)


class InputReader:

    def __init__(self, file_path_or_name, task_or_dataset=None, for_prompts=False, split=None):

        """
        file_path_or_name: path to the file containing examples.
                           Or can be a name of the dataset to load from
                           In this case, should use the split_to_load argument also
        task_or_dataset: which task or dataset the example is for
        split: what name to use for these examples
        """
        if isinstance(file_path_or_name, tuple):
            self.file_path_or_name = file_path_or_name[0]
            self.split_to_load = file_path_or_name[1]
        else:
            self.file_path_or_name = file_path_or_name
            self.split_to_load = None
        self.task_or_dataset = task_or_dataset
        self.for_prompts = for_prompts
        self.split = split
        self.predictions = None

        if self.split_to_load is not  None:
            self.dataset = self.load_dataset_from_hf(self.file_path_or_name, self.split_to_load, self.task_or_dataset, split)
        else:
            self.dataset = self.load_dataset_from_file(self.file_path_or_name, self.split, task=self.task_or_dataset)

    def load_dataset_from_hf(self, file_path_or_name, split_to_load, task, split_to_name):

        if file_path_or_name.lower() == 'mnli':
            print(file_path_or_name, split_to_load)
            data_hf = load_dataset('glue','mnli', split=split_to_load)
        else:
            print(file_path_or_name, split_to_load)
            data_hf = load_dataset(file_path_or_name, split=split_to_load,
                            cache_dir=None, use_auth_token=False)

        data = datasets.DatasetDict()
        data[split_to_name] = data_hf
        print(data)
        if task.lower() in ['snli', 'mnli']:
            data = self.map_columns_in_nli(data, split_to_name)
        print(data)
        return data

    def load_dataset_from_file(self, filename, split, task):

        # Decide which dataloader to use

        print(f'Loading dataset from file: {filename}')

        if 'csv' in filename:
            data = load_dataset("csv", data_files = {split:filename})
        elif 'tsv' in filename or 'txt' in filename: # assume txt file is also tsv file
            tsv_data = load_tsv_file_as_dict(filename)
            data_pd = datasets.Dataset.from_pandas(pd.DataFrame(data=tsv_data))
            data = datasets.DatasetDict()
            data[split] = data_pd
            print(data)
        elif 'jsonl' in filename:
            jsonl_data = load_jsonl_file(filename)
            data_pd = datasets.Dataset.from_pandas(pd.DataFrame(data=jsonl_data))
            data = datasets.DatasetDict()
            data[split] = data_pd
            print(data)
        elif 'json' in filename:
            data = load_dataset("json", data_files = {split:filename})


        if task.lower() in ['snli', 'mnli']:
            data = self.map_columns_in_nli(data, split)
        print(data)
        return data

    def write_predictions(self, output_data, output_dir, test_file):
        #print(self.dataset[self.split][:10])
        #print(output_data[:10])
        output_predict_file = os.path.join(output_dir, test_file.replace('/', '__') + ".preds.jsonl")
        print('Writing predictions to file ', output_predict_file)
        #self.predictions = predictions
        write_as_jsonl_file(output_predict_file, output_data)

    def map_columns_in_nli(self, dataset, split):

        if 'premise' not in dataset[split].column_names:
            if 'sentence1' in dataset[split].column_names:
                dataset[split] = dataset[split].add_column('premise', dataset[split]['sentence1'])
            elif 'Premise' in dataset[split].column_names:
                dataset[split] = dataset[split].add_column('premise', dataset[split]['Premise'])
        if 'hypothesis' not in dataset[split].column_names:
            if 'sentence2' in dataset[split].column_names:
                dataset[split] = dataset[split].add_column('hypothesis', dataset[split]['sentence2'])
            elif 'Hypothesis' in dataset[split].column_names:
                dataset[split] = dataset[split].add_column('hypothesis', dataset[split]['Hypothesis'])

        if 'premise' not in dataset[split].column_names and 'sentence1' not in dataset[split].column_names:
            if 'sentence1_binary_parse' in dataset[split].column_names:
                dataset[split] = dataset[split].add_column('premise', dataset[split]['sentence1_binary_parse'])

        if 'hypothesis' not in dataset[split].column_names and 'sentence2' not in dataset[split].column_names:
            if 'sentence2_binary_parse' in dataset[split].column_names:
                dataset[split] = dataset[split].add_column('hypothesis', dataset[split]['sentence2_binary_parse'])

        return dataset
