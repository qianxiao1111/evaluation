import os 
import json 
import pandas as pd
from inference import generate_outputs
from typing import List, Dict
from table_bench_eval.utils import read_json_file
from table_bench_eval.qa_metric import QAMetric
from table_bench_eval.utils import (
    parse_chart_code_then_exec, 
    parse_code_then_exec, 
    pre_save_table_to_csv,
    parse_final_answer_prediction,
    write_json_to_file,
    execution_eval
)

"""
Evaluation process modified from https://github.com/TableBench/TableBench
"""

def format_inputs(samples: List[Dict]) -> List:
    """
    输入数据格式化函数，按照 generate 的格式要求改造 inputs
    :param samples: 待格式化样例数据
    :param mode: 格式化模式
    """
    # 把需要推理的数据拼成 message 形式
    msgs = []
    for sample in samples:
        msg_sys = sample["instruction"]
        msg = [
            {"role": "system", "content": msg_sys},
        ]
        msgs.append(msg)
    return msgs

def model_infer_and_save(
    test_path, 
    llm_model, 
    tokenizer, 
    generate_args,
    inference_output_dir,
    base_model_name,
    n_samples_test=10, # for test mode, default None
):
    fnames = [x for x in os.listdir(test_path) if x.endswith('.jsonl')]
    all_samples = []
    for file_name in fnames:
        print(file_name)
        file_path = os.path.join(test_path, file_name)
        samples = read_json_file(file_path)
        if n_samples_test:
            samples = samples[:n_samples_test]
        msgs = format_inputs(samples)
        resp = generate_outputs(msgs, llm_model, tokenizer, generate_args)
        assert len(resp) == len(samples)
        for i, output in enumerate(resp):
            samples[i]["raw_generation"] = output["output_text"]
        save_path = os.path.join(inference_output_dir, base_model_name.split('/')[-1]+'_infer_'+file_name.split('.')[0]+'.jsonl')
        with open(save_path, 'w') as f:
            for item in samples:
                f.write(json.dumps(item)+'\n')
        all_samples.extend(samples)
    return all_samples

def execute_samples_and_save(all_samples, output_dir, base_model_name):
    DP_samples, TCOT_samples, SCoT_samples, Pot_samples = [], [], [], []
    for sample in all_samples:
        instruct_type = sample["instruction_type"]
        table = sample["table"]
        table = json.loads(table)
        pre_save_table_to_csv(table)
        prediction = sample["raw_generation"]
        qtype = sample['qtype']
        if "Final Answer" in prediction:
                parsed_prediction = parse_final_answer_prediction(prediction)
                parsed_result = {'parsed_prediction': parsed_prediction}
        else:
            if qtype == "Visualization":
                parsed_prediction, ecr_1 = parse_chart_code_then_exec(sample)
            else:
                parsed_prediction, ecr_1 = parse_code_then_exec(prediction)
            parsed_result = {
                'parsed_prediction': parsed_prediction, 'ecr_1': ecr_1
                }
                    # save parsed results
        if execution_eval(parsed_prediction):
            parsed_result['Parse@1'] = True
        else:
            parsed_result['Parse@1'] = False
        sample["parsed_result"] = parsed_result
        if instruct_type == "TCoT":
            TCOT_samples.append(sample)
        elif instruct_type == "SCoT":
            SCoT_samples.append(sample)
        elif instruct_type == "DP":
            DP_samples.append(sample)
        else:
            Pot_samples.append(sample)
    prompt_types = ["TCoT", "SCoT", "DP", "PoT"]
    prompt_samples = [TCOT_samples, SCoT_samples, DP_samples, Pot_samples]
    save_paths = [os.path.join(output_dir, base_model_name.split('/')[-1]+'_'+"execute_"+st+'.jsonl') for st in prompt_types]
    all_samples = []
    # save all the execute samples
    for save_path, samples in zip(save_paths, prompt_samples):
        all_samples.extend(samples)
        write_json_to_file(save_path, samples, is_json_line=True)

    return all_samples

def build_categoried_llm_inference_results(all_samples, base_model_name):
    '''
    categoried_llm_inference_results format is:
    {
        "model_name/prompt_type": { "merged_type": [result1, result2, ...] }
    }
    '''

    categoried_llm_inference_results = {}

    for result in all_samples:
        model_name = base_model_name
        type = result['qtype']
        subtype = result['qsubtype']
        merged_type = f'{type}_{subtype}'
        prompt_type = result['instruction_type']
        key = f'{model_name}/{prompt_type}'
        if key not in categoried_llm_inference_results:
            categoried_llm_inference_results[key] = {}
        if merged_type not in categoried_llm_inference_results[key]:
            categoried_llm_inference_results[key][merged_type] = []
        categoried_llm_inference_results[key][merged_type].append(result)

    return categoried_llm_inference_results

def eval_by_subtype(categoried_llm_inference_results, qa_metric, eval_result_dir, metric_name='ROUGE-L'):
    llm_eval_subtype_results = {}
    llm_eval_subtype_results_path = f'{eval_result_dir}/llm_eval_subtype_results.json'
    llm_eval_subtype_results_csv_path = f'{eval_result_dir}/llm_eval_subtype_results.csv'

    for key, result in categoried_llm_inference_results.items():
        print(f'Processing {key}...')
        for merge_type, results in result.items():
            if merge_type == 'Visualization_ChartGeneration':
                metric_scores = {
                    'F1': 0,
                    'EM': 0,
                    'ROUGE-L': 0,
                    'SacreBLEU': 0,
                }
                total = len(results)
                metric_scores['total'] = total
                ecr_1_acc = None
                ecr_1s = [result["parsed_result"].get('ecr_1', None)
                          for result in results]
                ecr_1_acc = ecr_1s.count(True) / total
                metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
                pass_1_acc = None
                parsed_prediction_results = []
                for result in results:
                    parsed_prediction = result["parsed_result"]['parsed_prediction']
                    if parsed_prediction == 'True':
                        parsed_prediction_results.append(True)
                    elif parsed_prediction == 'False':
                        parsed_prediction_results.append(False)
                    else:
                        parsed_prediction_results.append('None')
                pass_1_acc = parsed_prediction_results.count(True) / total
                metric_scores['Pass@1'] = round(pass_1_acc*100, 2)
            else:
                predictions = [result["parsed_result"]
                               ['parsed_prediction'] for result in results]
                references = [result['answer'] for result in results]
                metric_scores = qa_metric.compute(predictions, references)
                total = len(predictions)
                metric_scores['total'] = total
                if key.split('/')[1] == 'PoT':
                    ecr_1_acc = None
                    ecr_1s = [result["parsed_result"].get('ecr_1', None)
                              for result in results]
                    ecr_1_acc = ecr_1s.count(True) / total
                    metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
            parse_1s = [result["parsed_result"].get(
                'Parse@1', None) for result in results]
            metric_scores['Parse@1'] = round(
                parse_1s.count(True) / total * 100, 2)
            if key not in llm_eval_subtype_results:
                llm_eval_subtype_results[key] = {}
            llm_eval_subtype_results[key][merge_type] = metric_scores
    write_json_to_file(llm_eval_subtype_results_path, llm_eval_subtype_results)

    llm_eval_subtype_csv_results = []
    for key, result in llm_eval_subtype_results.items():
        csv_result = {
            'model_name': key.split('/')[0],
            'prompt_type': key.split('/')[1]
        }
        for merge_type, metric_scores in result.items():
            if merge_type == 'Visualization_ChartGeneration':
                csv_result[merge_type] = metric_scores['Pass@1']
            else:
                csv_result[merge_type] = metric_scores[metric_name]
        llm_eval_subtype_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_subtype_csv_results)
    llm_eval_df.to_csv(llm_eval_subtype_results_csv_path,
                       index=False, sep='\t')


def eval_by_type(categoried_llm_inference_results, qa_metric, eval_result_dir, metric_name='ROUGE-L'):
    llm_eval_type_results_path = f'{eval_result_dir}/llm_eval_type_results.json'
    llm_eval_type_results_csv_path = f'{eval_result_dir}/llm_eval_type_results.csv'
    llm_eval_type_results = {}
    for key, result in categoried_llm_inference_results.items():
        type_dict = {}
        print(f'Processing {key}...')
        for merge_type, results in result.items():
            type = merge_type.split('_')[0]
            if type not in type_dict:
                type_dict[type] = []
            type_dict[type].extend(results)
        for type, results in type_dict.items():
            if type == 'Visualization':
                metric_scores = {
                    'F1': 0,
                    'EM': 0,
                    'ROUGE-L': 0,
                    'SacreBLEU': 0,
                }
                total = len(results)
                metric_scores['total'] = total
                ecr_1_acc = None
                ecr_1s = [result["parsed_result"].get('ecr_1', None)
                          for result in results]
                ecr_1_acc = ecr_1s.count(True) / total
                metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
                pass_1_acc = None
                parsed_prediction_results = []
                for result in results:
                    parsed_prediction = result["parsed_result"]['parsed_prediction']
                    if parsed_prediction == 'True':
                        parsed_prediction_results.append(True)
                    elif parsed_prediction == 'False':
                        parsed_prediction_results.append(False)
                    else:
                        parsed_prediction_results.append('None')
                pass_1_acc = parsed_prediction_results.count(True) / total
                metric_scores['Pass@1'] = round(pass_1_acc*100, 2)
            else:
                predictions = [result["parsed_result"]['parsed_prediction']
                               for result in results]
                references = [result['answer'] for result in results]
                metric_scores = qa_metric.compute(predictions, references)
                total = len(predictions)
                metric_scores['total'] = total
                if key.split('/')[1] == 'PoT':
                    ecr_1_acc = None
                    ecr_1s = [result["parsed_result"].get('ecr_1', None)
                              for result in results]
                    ecr_1_acc = ecr_1s.count(True) / total
                    metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)
            parse_1s = [result["parsed_result"].get(
                'Parse@1', None) for result in results]
            metric_scores['Parse@1'] = round(
                parse_1s.count(True) / total * 100, 2)
            if key not in llm_eval_type_results:
                llm_eval_type_results[key] = {}
            llm_eval_type_results[key][type] = metric_scores
    write_json_to_file(llm_eval_type_results_path, llm_eval_type_results)

    llm_eval_type_csv_results = []
    for key, result in llm_eval_type_results.items():
        csv_result = {
            'model_name': key.split('/')[0],
            'prompt_type': key.split('/')[1]
        }
        for type, metric_scores in result.items():
            if type == 'Visualization':
                csv_result[type] = metric_scores['Pass@1']
            else:
                csv_result[type] = metric_scores[metric_name]
        llm_eval_type_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_type_csv_results)
    llm_eval_df.to_csv(llm_eval_type_results_csv_path, index=False, sep='\t')

def eval_by_overall(categoried_llm_inference_results, qa_metric, eval_result_dir, metric_name='ROUGE-L'):
    llm_eval_overall_results_path = f'{eval_result_dir}/llm_eval_overall_results.json'
    llm_eval_overall_results_csv_path = f'{eval_result_dir}/llm_eval_overall_results.csv'
    llm_eval_overall_results = {}
    for key, result in categoried_llm_inference_results.items():
        print(f'Processing {key}...')
        overall_results = []
        overall_wov_results = []
        overall_wv_results = []
        for merge_type, results in result.items():
            if merge_type == 'Visualization_ChartGeneration':
                overall_wv_results.extend(results)
            else:
                overall_wov_results.extend(results)
            overall_results.extend(results)
        metric_scores = {}
        total = len(overall_results)
        metric_scores['total'] = total

        wov_total = len(overall_wov_results)
        predictions = [result["parsed_result"]['parsed_prediction']
                       for result in overall_wov_results]
        references = [result['answer'] for result in overall_wov_results]
        wv_metric_scores = qa_metric.compute(predictions, references)
        rouge_l = wv_metric_scores['ROUGE-L']

        wv_total = len(overall_wv_results)
        parsed_predictions = [result["parsed_result"]['parsed_prediction']
                              for result in overall_wv_results]
        parsed_prediction_results = []
        for parsed_prediction in parsed_predictions:
            if parsed_prediction == 'True':
                parsed_prediction_results.append(True)
            elif parsed_prediction == 'False':
                parsed_prediction_results.append(False)
            else:
                parsed_prediction_results.append('None')
        if wv_total == 0:
            pass_1_acc = 0
        else:
            pass_1_acc = parsed_prediction_results.count(True) / wv_total * 100

        mix_metric = (rouge_l*wov_total + pass_1_acc*wv_total) / total
        metric_scores['MIX_Metric'] = round(mix_metric, 2)

        if key.split('/')[1] == 'PoT':
            ecr_1_acc = None
            ecr_1s = [result["parsed_result"].get('ecr_1', None)
                      for result in overall_results]
            ecr_1_acc = ecr_1s.count(True) / total
            metric_scores['ECR@1'] = round(ecr_1_acc*100, 2)

        parse_1s = [result["parsed_result"].get(
            'Parse@1', None) for result in overall_results]
        metric_scores['Parse@1'] = round(
            parse_1s.count(True) / total * 100, 2)
        llm_eval_overall_results[key] = metric_scores
    write_json_to_file(llm_eval_overall_results_path, llm_eval_overall_results)

    llm_eval_overall_csv_results = []
    for key, metric_scores in llm_eval_overall_results.items():
        csv_result = {
            'model_name': key.split('/')[0],
            'prompt_type': key.split('/')[1]
        }
        csv_result['overall'] = metric_scores['MIX_Metric']
        csv_result['Parse@1'] = metric_scores['Parse@1']
        llm_eval_overall_csv_results.append(csv_result)
    llm_eval_df = pd.DataFrame(llm_eval_overall_csv_results)
    llm_eval_df.to_csv(llm_eval_overall_results_csv_path,
                       index=False, sep='\t')

        
def run_eval(all_samples, output_dir, base_model_name):
    eval_result_dir = os.path.join(output_dir, "eval_results")
    if not os.path.exists(eval_result_dir):
        os.mkdir(eval_result_dir)
    metric_name = 'ROUGE-L'
    qa_metric = QAMetric()

    categoried_llm_inference_results = build_categoried_llm_inference_results(all_samples, base_model_name)

    print('-'*10, 'Eval by subtype', '-'*10)
    eval_by_subtype(categoried_llm_inference_results, qa_metric, eval_result_dir, metric_name)
    print('-'*10, 'Eval by type', '-'*10)
    eval_by_type(categoried_llm_inference_results, qa_metric, eval_result_dir, metric_name)
    print('-'*10, 'Eval by overall', '-'*10)
    eval_by_overall(categoried_llm_inference_results, qa_metric, eval_result_dir, metric_name)





    





        
