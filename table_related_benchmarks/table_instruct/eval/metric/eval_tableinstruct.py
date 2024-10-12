
import json
from table_instruct.eval.scripts.table_utils import evaluate as table_llama_eval
from table_instruct.eval.scripts.metric import *
from rouge_score import rouge_scorer
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
from nltk.translate import meteor_score
import time

def eval_hitab_ex(data):
    pred_list = []
    gold_list = []
    for i in range(len(data)):
        if len(data[i]["predict"].strip("</s>").split(">, <")) > 1:
            instance_pred_list = data[i]["predict"].strip("</s>").split(">, <")
            pred_list.append(instance_pred_list)
            gold_list.append(data[i]["output"].strip("</s>").split(">, <"))
        else:
            pred_list.append(data[i]["predict"].strip("</s>"))
            gold_list.append(data[i]["output"].strip("</s>"))
    result=table_llama_eval(gold_list, pred_list)
    return result

def compute_rouge(list1, list2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for sent1, sent2 in zip(list1, list2):
        score = scorer.score(sent1, sent2)
        scores.append(score)
    rouge1 = np.mean([score['rouge1'].fmeasure for score in scores])
    rouge2 = np.mean([score['rouge2'].fmeasure for score in scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in scores])
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

def compute_bleu(list1, list2):
    bleu_scores = []
    smoothie = SmoothingFunction().method4  # 用于平滑处理BLEU分数
    for ref, pred in zip(list1, list2):
        reference = [ref.split()]  # BLEU 接受参考文本列表
        candidate = pred.split()
        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(score)
    bleu_score = np.mean(bleu_scores)
    return bleu_score


def compute_sacrebleu(reference_list, candidate_list):
    individual_scores = []

    for ref, pred in zip(reference_list, candidate_list):
        # 计算每对句子的 BLEU 分数
        score = sacrebleu.sentence_bleu(pred, [ref])  # 参考文本需要是列表形式
        individual_scores.append(score.score)

    # 计算平均分
    average_bleu = sum(individual_scores) / len(individual_scores)
    return average_bleu


def compute_meteor(reference_list, candidate_list):
    individual_scores = []

    for ref, pred in zip(reference_list, candidate_list):
        ref_tokens = ref.split()  # 参考句子分词
        pred_tokens = pred.split()  # 预测句子分词

        # 直接传入已分词的列表
        score = meteor_score.single_meteor_score(ref_tokens, pred_tokens)
        individual_scores.append(score)

    # 计算平均分
    average_meteor = sum(individual_scores) / len(individual_scores)
    return average_meteor


def eval_bleu(data):
    test_examples_answer = [x["output"] for x in data]
    test_predictions_pred = [x["predict"].strip("</s>") for x in data]
    predictions = test_predictions_pred
    references = test_examples_answer

    #rouge = evaluate.load('rouge')
    #result_rouge = rouge.compute(predictions=predictions, references=references)
    result_rouge = compute_rouge(references,predictions)
    result_bleu = compute_bleu(references,predictions)
    result_sacrebleu = compute_sacrebleu(references,predictions)
    result_meteor = compute_meteor(references,predictions)

    result = {
        'rouge':result_rouge,
        'bleu':result_bleu,
        'sacrebleu':result_sacrebleu,
        'meteor':result_meteor
    }
    return result

def eval_ent_link_acc(data):
    #assert len(data) == 2000
    correct_count = 0
    multi_candidates_example_count = 0
    for i in range(len(data)):
        candidate_list = data[i]["candidates_entity_desc_list"]
        ground_truth = data[i]["output"].strip("<>").lower()
        predict = data[i]["predict"][:-4].strip("<>").lower()
        # import pdb
        # pdb.set_trace()

        if ground_truth == predict:
            correct_count += 1
        if len(candidate_list) > 1:
            multi_candidates_example_count += 1

    acc=correct_count / len(data)
    result={
        "correct_count":correct_count,
        "acc":acc
    }
    return result

def eval_col_pop_map(data):
    rs = []
    recall = []
    for i in range(len(data)):
        ground_truth = data[i]["target"].strip(".")
        # ground_truth = data[i]["target"].strip(".")
        pred = data[i]["predict"].strip(".")
        if "</s>" in pred:
            end_tok_ix = pred.rfind("</s>")
            pred = pred[:end_tok_ix]
        ground_truth_list = ground_truth.split(", ")
        # ground_truth_list = test_col_pop_rank[i]["target"].strip(".").split(", ")
        pred_list = pred.split(", ")
        for k in range(len(pred_list)):
            pred_list[k] = pred_list[k].strip("<>")

        # print(len(ground_truth_list), len(pred_list))

        # import pdb
        # pdb.set_trace()
        # add to remove repeated generated item
        new_pred_list = list(set(pred_list))
        new_pred_list.sort(key=pred_list.index)
        r = [1 if z in ground_truth_list else 0 for z in new_pred_list]
        ap = average_precision(r)
        # print("ap:", ap)
        rs.append(r)

        # if sum(r) != 0:
        #     recall.append(sum(r)/len(ground_truth_list))
        # else:
        #     recall.append(0)
        recall.append(sum(r) / len(ground_truth_list))
    map = mean_average_precision(rs)
    m_recall = sum(recall) / len(data)
    if map + m_recall == 0:
        f1=0
    else:
        f1 = 2 * map * m_recall / (map + m_recall)
    #print(data_name, len(data))
    #print("mean_average_precision:", map)
    result={
        "mean_average_precision":map,
        "mean_average_recall":m_recall,
        "f1":f1
    }
    return result

def eval_col_type_f1(data):
    #rel_ex也用这一套
    ground_truth_list = []
    pred_list = []
    for i in range(len(data)):
        item = data[i]
        ground_truth = item["ground_truth"]
        # pred = item["predict"].strip("</s>").split(",")
        pred = item["predict"].split("</s>")[0].split(", ")
        ground_truth_list.append(ground_truth)
        pred_list.append(pred)

    total_ground_truth_col_types = 0
    total_pred_col_types = 0
    joint_items_list = []
    for i in range(len(ground_truth_list)):
        total_ground_truth_col_types += len(ground_truth_list[i])
        total_pred_col_types += len(pred_list[i])
        joint_items = [item for item in pred_list[i] if item in ground_truth_list[i]]
        joint_items_list += joint_items

    # import pdb
    # pdb.set_trace()

    gt_entire_col_type = {}
    for i in range(len(ground_truth_list)):
        gt = list(set(ground_truth_list[i]))
        for k in range(len(gt)):
            if gt[k] not in gt_entire_col_type.keys():
                gt_entire_col_type[gt[k]] = 1
            else:
                gt_entire_col_type[gt[k]] += 1
    # print(len(gt_entire_col_type.keys()))

    pd_entire_col_type = {}
    for i in range(len(pred_list)):
        pd = list(set(pred_list[i]))
        for k in range(len(pd)):
            if pd[k] not in pd_entire_col_type.keys():
                pd_entire_col_type[pd[k]] = 1
            else:
                pd_entire_col_type[pd[k]] += 1
    # print(len(pd_entire_col_type.keys()))

    joint_entire_col_type = {}
    for i in range(len(joint_items_list)):
        if joint_items_list[i] not in joint_entire_col_type.keys():
            joint_entire_col_type[joint_items_list[i]] = 1
        else:
            joint_entire_col_type[joint_items_list[i]] += 1
    # print(len(joint_entire_col_type.keys()))

    precision = len(joint_items_list) / total_pred_col_types
    recall = len(joint_items_list) / total_ground_truth_col_types
    if precision + recall==0:
        f1=0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    sorted_gt = sorted(gt_entire_col_type.items(), key=lambda x: x[1], reverse=True)

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return result

def eval_tabfact_acc(data):
    correct = 0
    remove_count = 0
    for i in range(len(data)):
        ground_truth = data[i]["output"]
        prediction = data[i]["predict"].strip("</s>")
        # if prediction.find(ground_truth) == 0:
        if prediction == ground_truth:
            correct += 1
        if prediction.find("<s>") == 0:
            remove_count += 1

    acc=correct / (len(data) - remove_count)
    result={
        "correct":correct,
        "accuracy":acc
    }
    return result

def eval_row_pop_map(data):
    rs = []
    recall = []
    ap_list = []
    for i in range(len(data)):
        pred = data[i]["predict"].strip(".")
        if "</s>" in pred:
            end_tok_ix = pred.rfind("</s>")
            pred = pred[:end_tok_ix]
        ground_truth_list = data[i]["target"]
        pred_list = pred.split(", ")
        for k in range(len(pred_list)):
            pred_list[k] = pred_list[k].strip("<>")

        # add to remove repeated generated item
        new_pred_list = list(set(pred_list))
        new_pred_list.sort(key=pred_list.index)
        # r = [1 if z in ground_truth_list else 0 for z in pred_list]
        r = [1 if z in ground_truth_list else 0 for z in new_pred_list]
        # ap = average_precision(r)
        ap = row_pop_average_precision(r, ground_truth_list)
        # print("ap:", ap)
        ap_list.append(ap)

    map = sum(ap_list) / len(data)
    m_recall = sum(recall) / len(data)
    if map + m_recall == 0:
        f1 = 0
    else:
        f1 = 2 * map * m_recall / (map + m_recall)
    # print(data_name, len(data))
    # print("mean_average_precision:", map)
    result = {
        "mean_average_precision": map,
        "mean_average_recall": m_recall,
        "f1": f1
    }
    return result