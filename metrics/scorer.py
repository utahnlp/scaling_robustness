import os, sys, json, csv
import evaluate
from pprint import pprint
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

def load_jsonl_file(filename):

    data = []
    with open(filename, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))


    return data

class AccuracyScorer:

    def __init__(self, filename=None, gold_key=None, pred_key=None, pred_label_map=None, gold_label_map=None, group_by=None, verbose=False):

        self.filename = filename
        self.metric_name = 'accuracy'
        self.metric_fn = self.calculate_accuracy
        self.verbose = verbose

        self.group_by = group_by
        self.gold_key = gold_key
        self.pred_key = pred_key
        self.pred_label_map = pred_label_map
        self.gold_label_map = gold_label_map

        self.golds, self.preds, self.other_details = self.load_from_file(self.filename)
        self.golds, self.preds = self.preprocess(self.golds, self.preds)

    def calculate(self):

        # overall metric
        self.metric = self.metric_fn(self.golds, self.preds)

        # group examples
        self.group_by_dict = self.examples_group_by(self.other_details)

        # group metrics
        self.group_by_metric = self.calculate_metrics_by_group(self.golds, self.preds, self.group_by_dict)

    def calculate_metrics_by_group(self, golds, preds, group_by_dict):

        if len(group_by_dict) <2:
            return {}

        group_by_metric = {}
        for key, val in group_by_dict.items():
            group_golds = [golds[i] for i in val]
            group_preds = [preds[i] for i in val]
            if len(group_golds) == 0:
                continue
            group_by_metric[key] = self.metric_fn(group_golds, group_preds)

        return group_by_metric

    def examples_group_by(self, other_details):

        if self.group_by not in other_details[0]:
            return {}

        group_by_dict = {}
        none_found = 0.0

        total = []
        for i, detail in enumerate(other_details):
            cat = detail[self.group_by]
            if detail[self.group_by] is None:
                none_found +=1
                continue
            total.append(i)
            if ';' in cat.lower():
                cats = cat.lower().split(';')
            else:
                cats = [cat.lower()]

            for cat in cats:
                if cat not in group_by_dict:
                    group_by_dict[cat] = []
                group_by_dict[cat].append(i)
        if self.verbose:
            for key, val in group_by_dict.items():
                print(f'For {key}, we have {len(val)} examples.')

        # to calculate global metric for non-None values
        #group_by_dict[self.group_by] = total
        if self.verbose:
            print('Total examples (exclude null/None) ', len(total))
            print('With nul//None value ', none_found)

        return group_by_dict

    def load_from_file(self, filename):
        if filename.endswith('.jsonl'):
            examples = load_jsonl_file(filename)

        assert self.gold_key is not None
        assert self.pred_key is not None

        golds = []
        preds = []
        other_details = []
        for ex in examples:
            golds.append(ex[self.gold_key])
            preds.append(ex[self.pred_key])
            d = {}
            for k in ex.keys():
                if k not in [self.gold_key, self.pred_key]:
                    d[k] = ex[k]
            other_details.append(d)
        return golds, preds, other_details

    def preprocess(self, golds, preds):

        if self.pred_label_map is not None:
            new_preds = [self.pred_label_map[p] for p in preds]
        else:
            new_preds = preds

        if self.gold_label_map is not None:
            new_golds = [self.gold_label_map[g] for g in golds]
        else:
            new_golds = golds

        return new_golds, new_preds

    def calculate_accuracy(self, golds, preds):

        #accuracy_metric = evaluate.load("accuracy")
        #results = accuracy_metric.compute(references=golds, predictions=preds)
        count = 0.0
        total = 0.0
        if isinstance(golds[0], str):
            golds = [g.lower().strip() for g in golds]
            preds = [p.lower().strip() for p in preds]
        for i in range(len(golds)):
            if golds[i] == preds[i]:
                count +=1
            total +=1

        acc = count / total
        return acc

class F1Scorer:

    def __init__(self, filename=None, gold_key=None, pred_key=None, pred_label_map=None, gold_label_map=None, group_by=None, verbose=False):

        self.filename = filename
        self.metric_name = 'f1'
        self.metric_fn = f1_score
        self.verbose = verbose

        self.group_by = group_by
        self.gold_key = gold_key
        self.pred_key = pred_key
        self.pred_label_map = pred_label_map
        self.gold_label_map = gold_label_map

        self.golds, self.preds, self.other_details = self.load_from_file(self.filename)
        self.golds, self.preds = self.preprocess(self.golds, self.preds)

    def calculate(self):

        # overall metric
        if self.metric_name == 'f1':
            self.metric = self.metric_fn(y_true=self.golds, y_pred=self.preds)
        else:
            self.metric = self.metric_fn(self.golds, self.preds)

        # group examples
        self.group_by_dict = self.examples_group_by(self.other_details)

        # group metrics
        self.group_by_metric = self.calculate_metrics_by_group(self.golds, self.preds, self.group_by_dict)

    def calculate_metrics_by_group(self, golds, preds, group_by_dict):

        if len(group_by_dict) <2:
            return {}

        group_by_metric = {}
        for key, val in group_by_dict.items():
            group_golds = [golds[i] for i in val]
            group_preds = [preds[i] for i in val]
            if len(group_golds) == 0:
                continue
            group_by_metric[key] = self.metric_fn(group_golds, group_preds)

        return group_by_metric

    def examples_group_by(self, other_details):

        if self.group_by not in other_details[0]:
            return {}

        group_by_dict = {}
        none_found = 0.0

        total = []
        for i, detail in enumerate(other_details):
            cat = detail[self.group_by]
            if detail[self.group_by] is None:
                none_found +=1
                continue
            total.append(i)
            if ';' in cat.lower():
                cats = cat.lower().split(';')
            else:
                cats = [cat.lower()]

            for cat in cats:
                if cat not in group_by_dict:
                    group_by_dict[cat] = []
                group_by_dict[cat].append(i)
        if self.verbose:
            for key, val in group_by_dict.items():
                print(f'For {key}, we have {len(val)} examples.')

        # to calculate global metric for non-None values
        #group_by_dict[self.group_by] = total
        if self.verbose:
            print('Total examples (exclude null/None) ', len(total))
            print('With nul//None value ', none_found)

        return group_by_dict

    def load_from_file(self, filename):
        if filename.endswith('.jsonl'):
            examples = load_jsonl_file(filename)

        assert self.gold_key is not None
        assert self.pred_key is not None

        golds = []
        preds = []
        other_details = []
        for ex in examples:
            golds.append(ex[self.gold_key])
            preds.append(ex[self.pred_key])
            d = {}
            for k in ex.keys():
                if k not in [self.gold_key, self.pred_key]:
                    d[k] = ex[k]
            other_details.append(d)
        return golds, preds, other_details

    def preprocess(self, golds, preds):

        if self.pred_label_map is not None:
            new_preds = [self.pred_label_map[p] for p in preds]
        else:
            new_preds = preds

        if self.gold_label_map is not None:
            new_golds = [self.gold_label_map[g] for g in golds]
        else:
            new_golds = golds

        return new_golds, new_preds

    def calculate_accuracy(self, golds, preds):

        #accuracy_metric = evaluate.load("accuracy")
        #results = accuracy_metric.compute(references=golds, predictions=preds)
        count = 0.0
        total = 0.0
        if isinstance(golds[0], str):
            golds = [g.lower().strip() for g in golds]
            preds = [p.lower().strip() for p in preds]
        for i in range(len(golds)):
            if golds[i] == preds[i]:
                count +=1
            total +=1

        acc = count / total
        return acc

#class NLIDiagScorer(AccuracyScorer):
#
#    def __init__(self, metric_to_use='matthews_corrcoef', **kwargs):
#
#        super(NLIDiagScorer, self).__init__(**kwargs)
#        self.metric_to_use = metric_to_use
#        self.metric_fn = matthews_corrcoef
#
#    def calculate(self):
#        for category in ['Lexical Semantics', 'Predicate-Argument Structure', 'Logic', 'Knowledge', 'Domain']:
#            nli_diag = NLIDiagScorer(filename=self.filename, gold_key='label', pred_key='predicted', group_by=category)
#            nli_diag.calculate()

class NLIDiagScorer:

    def __init__(self, metric_to_use='matthews_corrcoef', **kwargs):

        self.metric_to_use = metric_to_use
        self.metric_name = self.metric_to_use
        self.metric_fn = matthews_corrcoef
        self.kwargs = kwargs
        self.metric = {}
        self.metric_accuracy = {}

    def calculate(self):
        for category in ['Lexical Semantics', 'Predicate-Argument Structure', 'Logic', 'Knowledge', 'Domain']:
            #nli_diag = AccuracyScorer(filename=self.filename, gold_key='label', pred_key='predicted', group_by=category)
            nli_diag = AccuracyScorer(group_by=category, **self.kwargs)
            nli_diag.metric_fn = self.metric_fn
            nli_diag.calculate()
            for sub_cat, met in nli_diag.group_by_metric.items():
                key = (category, sub_cat)
                self.metric[key] = met
            nli_diag = AccuracyScorer(group_by=category, **self.kwargs)
            nli_diag.metric_fn = nli_diag.calculate_accuracy
            nli_diag.calculate()
            for sub_cat, met in nli_diag.group_by_metric.items():
                key = (category, sub_cat)
                self.metric_accuracy[key] = met

class NLIDiagScorerAccuracy:

    def __init__(self, metric_to_use='matthews_corrcoef', **kwargs):

        self.metric_to_use = metric_to_use
        self.metric_name = "accuracy"
        #self.metric_fn = matthews_corrcoef
        self.kwargs = kwargs
        self.metric = {}
        self.metric_accuracy = {}

    def calculate(self):
        for category in ['Lexical Semantics', 'Predicate-Argument Structure', 'Logic', 'Knowledge', 'Domain']:
            #nli_diag = AccuracyScorer(filename=self.filename, gold_key='label', pred_key='predicted', group_by=category)
            nli_diag = AccuracyScorer(group_by=category, **self.kwargs)
            nli_diag.calculate()
            for sub_cat, met in nli_diag.group_by_metric.items():
                key = (category, sub_cat)
                self.metric[key] = met
            #nli_diag = AccuracyScorer(group_by=category, **self.kwargs)
            #nli_diag.metric_fn = nli_diag.calculate_accuracy
            #nli_diag.calculate()
            #for sub_cat, met in nli_diag.group_by_metric.items():
            #    key = (category, sub_cat)
            #    self.metric_accuracy[key] = met


class HansScorer:

    def __init__(self, metric_to_use='accuracy', **kwargs):

        self.metric_to_use = metric_to_use
        self.metric_name = 'accuracy'
        self.kwargs = kwargs
        self.metric = {}

    def calculate(self):
        for category in ['heuristic']:
            hans = AccuracyScorer(group_by=category, **self.kwargs)
            #hans.metric_fn = self.metric_fn
            hans.calculate()
            for sub_cat, met in hans.group_by_metric.items():
                key = (category, sub_cat)
                self.metric[key] = met

class AucQQP:

    def __init__(self, filename=None, gold_key=None, pred_key=None, pred_label_map=None, gold_label_map=None, group_by=None, verbose=False):

        self.filename = filename
        self.metric_name = 'roc_auc'
        self.metric_fn = roc_auc_score
        self.verbose = verbose

        self.group_by = group_by
        self.gold_key = gold_key
        self.pred_key = pred_key
        self.pred_label_map = pred_label_map
        self.gold_label_map = gold_label_map

        self.golds, self.preds, self.other_details = self.load_from_file(self.filename)
        self.golds, self.preds = self.preprocess(self.golds, self.preds)

        self.labels, self.probs = self.convert_to_sklearn_format(self.other_details, self.golds)
        #print(self.labels, self.probs)

    def calculate(self):

        score = self.metric_fn(self.labels, self.probs)

        self.metric = score

    def convert_to_sklearn_format(self, other_details, golds):
        # treat not_duplicate as 0, duplicate as 1
        # collect probs for duplicate
        label_map = {'not_duplicate': 0, 'duplicate': 1}
        #label_map = {'not_duplicate': 1, 'duplicate': 0}
        labels = []
        probs = []
        for i in range(len(golds)):
            lb = label_map[golds[i]]
            labels.append(lb)
            prob = self.other_details[i]['probs']['duplicate']
            #prob = self.other_details[i]['probs']['not_duplicate']
            probs.append(prob)
        return labels, probs

    def load_from_file(self, filename):
        if filename.endswith('.jsonl'):
            examples = load_jsonl_file(filename)

        assert self.gold_key is not None
        assert self.pred_key is not None

        golds = []
        preds = []
        other_details = []
        for ex in examples:
            golds.append(ex[self.gold_key])
            preds.append(ex[self.pred_key])
            d = {}
            for k in ex.keys():
                if k not in [self.gold_key, self.pred_key]:
                    d[k] = ex[k]
            other_details.append(d)
        return golds, preds, other_details

    def preprocess(self, golds, preds):

        if self.pred_label_map is not None:
            new_preds = [self.pred_label_map[p] for p in preds]
        else:
            new_preds = preds

        if self.gold_label_map is not None:
            new_golds = [self.gold_label_map[g] for g in golds]
        else:
            new_golds = golds

        return new_golds, new_preds

class AucMRPC:

    def __init__(self, filename=None, gold_key=None, pred_key=None, pred_label_map=None, gold_label_map=None, group_by=None, verbose=False):

        self.filename = filename
        self.metric_name = 'roc_auc'
        self.metric_fn = roc_auc_score
        self.verbose = verbose

        self.group_by = group_by
        self.gold_key = gold_key
        self.pred_key = pred_key
        self.pred_label_map = pred_label_map
        self.gold_label_map = gold_label_map

        self.golds, self.preds, self.other_details = self.load_from_file(self.filename)
        self.golds, self.preds = self.preprocess(self.golds, self.preds)

        self.labels, self.probs = self.convert_to_sklearn_format(self.other_details, self.golds)
        #print(self.labels, self.probs)

    def calculate(self):

        score = self.metric_fn(self.labels, self.probs)

        self.metric = score

    def convert_to_sklearn_format(self, other_details, golds):
        # treat not_equivalent as 0, equivalent as 1
        # collect probs for equivalent
        label_map = {'not_equivalent': 0, 'equivalent': 1}
        #label_map = {'not_equivalent': 1, 'equivalent': 0}
        labels = []
        probs = []
        for i in range(len(golds)):
            lb = label_map[golds[i]]
            labels.append(lb)
            prob = self.other_details[i]['probs']['equivalent']
            #prob = self.other_details[i]['probs']['not_equivalent']
            probs.append(prob)
        return labels, probs

    def load_from_file(self, filename):
        if filename.endswith('.jsonl'):
            examples = load_jsonl_file(filename)

        assert self.gold_key is not None
        assert self.pred_key is not None

        golds = []
        preds = []
        other_details = []
        for ex in examples:
            golds.append(ex[self.gold_key])
            preds.append(ex[self.pred_key])
            d = {}
            for k in ex.keys():
                if k not in [self.gold_key, self.pred_key]:
                    d[k] = ex[k]
            other_details.append(d)
        return golds, preds, other_details

    def preprocess(self, golds, preds):

        if self.pred_label_map is not None:
            new_preds = [self.pred_label_map[p] for p in preds]
        else:
            new_preds = preds

        if self.gold_label_map is not None:
            new_golds = [self.gold_label_map[g] for g in golds]
        else:
            new_golds = golds

        return new_golds, new_preds


# implement a group by feature where you provide the list of keys as input and their tuple will be used to group the examples, and then the accuracy is predicted for each of those key tuples

if __name__=="__main__":

    #hans = AccuracyScorer(filename="../final_models/mnli/t5-3b/seed-1/evaluations/robustness__hans__hans__heuristics_evaluation_set.jsonl.preds.jsonl", gold_key='gold_label', pred_key='predicted', pred_label_map={'entailment': 'entailment', 'neutral': 'non-entailment', 'contradiction': 'non-entailment'}, group_by='heuristic')
    #hans.calculate()
    #print(hans.metric)
    #pprint(hans.group_by_metric)

    #nli_diag = AccuracyScorer(filename="../final_models/mnli/t5-3b/seed-1/evaluations/robustness__nli-diagnostics__diagnostic-full.csv.preds.jsonl", gold_key='label', pred_key='predicted', group_by='Lexical Semantics')
    #nli_diag.calculate()
    #print(nli_diag.metric)
    #pprint(nli_diag.group_by_metric)

    #for category in ['Lexical Semantics', 'Predicate-Argument Structure', 'Logic', 'Knowledge', 'Domain']:
    #    print('=='*20)
    #    print('Calculating metrics for ', category)
    #    nli_diag = NLIDiagScorer(filename="../final_models/mnli/t5-11b/seed-1/evaluations/robustness__nli-diagnostics__diagnostic-full.csv.preds.jsonl", gold_key='label', pred_key='predicted', group_by=category)
    #    nli_diag.calculate()
    #    print(nli_diag.metric)
    #    pprint(nli_diag.group_by_metric)
    #    print('=='*20)

    # following is the working syntax
    #nli_diag = NLIDiagScorer(filename="../final_models/mnli/t5-11b/seed-1/evaluations/robustness__nli-diagnostics__diagnostic-full.csv.preds.jsonl", gold_key='label', pred_key='predicted')
    #nli_diag.calculate()
    #pprint(nli_diag.metric)


    #hans = HansScorer(filename="../final_models/mnli/t5-3b/seed-1/evaluations/robustness__hans__hans__heuristics_evaluation_set.jsonl.preds.jsonl", gold_key='gold_label', pred_key='predicted', pred_label_map={'entailment': 'entailment', 'neutral': 'non-entailment', 'contradiction': 'non-entailment'})
    #hans.calculate()
    #pprint(hans.metric)
    input_file = sys.argv[1]

    #f1 = F1Scorer(filename=input_file, pred_key='predicted', gold_key='label', pred_label_map={'not_duplicate': 0, 'duplicate': 1}, gold_label_map={'not_duplicate': 0, 'duplicate': 1})
    #f1.calculate()
    #print(f1.metric)

    auc = AucQQP(filename=input_file, pred_key='predicted', gold_key='label')
    auc.calculate()
    print(auc.metric)

    auc = AccuracyScorer(filename=input_file, pred_key='predicted', gold_key='label')
    auc.calculate()
    print(auc.metric)



    #auc = AucMRPC(filename=input_file, pred_key='predicted', gold_key='label', gold_label_map = {'0': 'not_equivalent', '1': 'equivalent'})
    #auc.calculate()
    #print(auc.metric)


    #auc = AccuracyScorer(filename=input_file, pred_key='predicted', gold_key='label', gold_label_map = {'0': 'not_equivalent', '1': 'equivalent'})
    #auc.calculate()
    #print(auc.metric)
