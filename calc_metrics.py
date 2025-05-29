import math
import argparse
import collections
import json
from tqdm import tqdm
from tabulate import tabulate
from model.eval.eval_utils import *
from model.eval.glossary import *
from model.eval.evaluate_metrics import calculate_bleu, calculate_exactmatch, calculate_f1score, similarity_candidate_prediction, calculate_appearance_with_normalization
from nltk.translate.bleu_score import sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument('--pred', type=str, default="answer-file-medplib-zeorshot.jsonl", help='path to prediction file')


def evaluate(test_dict_lst, args, dataset):
    print('pred file', args.pred)

    candidate_set = []
    for item in test_dict_lst:
        candidate_set.append(item['gt'])
    
    print(len(candidate_set))
    candidate_set = set(candidate_set)
    print('candidate_set',len(candidate_set))

    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    open_hit_scores = collections.defaultdict(list)

    eval_open = False
    eval_closed = False
    open_cnt = 0
    closed_cnt = 0
    for item in tqdm(test_dict_lst):
        gt_value = str(item['gt']).lower()
        pred_value = item['output'].lower()


        if item['gt'] == 'A':
            # avoid 'a' as a ground truth
            gt_value = 'a'
        else:
            # normalize the ground truth value
            gt_value = normalize_word(gt_value)
        if item['output'] == 'A' or item['output'].startswith('A,'):
            # avoid 'a' as a prediction
            pred_value = 'a'
        else:
            pred_value = normalize_word(pred_value)

        if 'VQA-RAD' in dataset:
            if gt_value in ['yes', 'no']:
                eval_closed = True
                dataset = 'VQA-RAD-yesno'
            else:
                eval_open = True
                dataset = 'VQA-RAD-open'

        if dataset in ['MeCoVQA', 'VQA-RAD-open']:
            eval_open = True
            # for open-ended question
            # if gt_value in pred_value:
            #     hit = 1.0
            # else:
            #     hit = 0.0
            # open_hit_scores['hit'].append(hit)

            print('##pred_value', pred_value, '##gt_value', gt_value)
            print(calculate_appearance_with_normalization(pred_value, gt_value, candidate_set))

            open_hit_scores['hit'].append(calculate_appearance_with_normalization(pred_value, gt_value, candidate_set))
            open_hit_scores['q_id'].append(item['id'])

            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(item['id'])

            # import pdb; pdb.set_trace()

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(item['id'])

            # if isinstance(f1_scores['hit'][-1], str):
            #     # import pdb; pdb.set_trace()

            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
            
            bleu_scores['q_id'].append(item['id'])
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)
            open_cnt += 1

        elif dataset in ['PMC-VQA', 'OmniMedVQA', 'VQA-RAD-yesno']:
            eval_closed = True
            # for close-ended question (Yes/No)
            closed_scores['q_id'].append(item['id'])
            # if 'yes' in pred_value or 'no' in pred_value:
            #     if gt_value in pred_value:
            #         closed_scores['hit'].append(1)
            #     else:
            #         closed_scores['hit'].append(0)
            # else:
            #     closed_scores['hit'].append(0)


            if dataset == 'VQA-RAD-yesno':
                pred_value_filtered = filter_closed_answers(pred_value, 'yes/no')
                gt_option = gt_value.lower().strip()
            else:
                pred_value_filtered = filter_closed_answers(pred_value, 'multiple_choice')
                # extract answer value for multiple choice questions
                gt_option = extract_answer_option(item['input'], gt_value)

            # print('pred_value', pred_value, 'gt_value', gt_value)
            # print('##gt_option', gt_option, '##pred_value', pred_value)
            if gt_value == pred_value_filtered or gt_option in pred_value:
                closed_scores['hit'].append(1)
            else:
                print('##pred_value', pred_value, '##gt_value', gt_value)
                closed_scores['hit'].append(0)
            closed_cnt += 1
    
    print('open_cnt', open_cnt)
    print('closed_cnt', closed_cnt)
    if eval_open:
        # import pdb; pdb.set_trace()

        print(sum(open_hit_scores['hit']),len(open_hit_scores['hit']), len(open_hit_scores['q_id']))
        open_hit_score = sum(open_hit_scores['hit']) / len(open_hit_scores['hit'])
        exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
        f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
        precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
        recall = sum(f1_scores['recall']) / len(f1_scores['recall'])

        bleu_score   = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score'])
        bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1'])
        bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2'])
        bleu_score_3 = sum(bleu_scores['bleu_score_3']) / len(bleu_scores['bleu_score_3'])
    else:
        exact_score = 0.0
        f1_score = 0.0
        precision = 0.0
        recall = 0.0
        open_hit_score = 0.0
        bleu_score   = 0.0
        bleu_score_1 = 0.0
        bleu_score_2 = 0.0
        bleu_score_3 = 0.0

    if eval_closed:
        closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0
    else:
        closed_score = 0.0

    num_close, num_open = len(closed_scores['hit']), len(exact_scores['q_id'])
    print(f'num_open {num_open} || num_close {num_close}')

    return tabulate(
        [
            ['exact match acc score', exact_score*100], 
            ['f1 score', f1_score*100], 
            ['precision', precision*100], 
            ['recall', recall*100], 
            ['bleu_score', bleu_score*100], 
            ['bleu_score_1', bleu_score_1*100], 
            ['bleu_score_2', bleu_score_2*100], 
            ['bleu_score_3', bleu_score_3*100], 
            ['open accuracy', open_hit_score*100],
            ['yes/no accuracy', closed_score*100]
        ], 
        headers=['Metric', 'Performance']
    )

if __name__ == "__main__":
    args = parser.parse_args()

    test_dict_lst = json.load(open(args.pred, 'r'))['outputs']

    print('test_dict_lst', len(test_dict_lst))
    dataset = args.pred.split('/')[-2].split('_')[3]

    if len(test_dict_lst) == 0:
        print('No predictions found in the file.')
    else:
        result = evaluate(test_dict_lst, args, dataset)
        print(result)
