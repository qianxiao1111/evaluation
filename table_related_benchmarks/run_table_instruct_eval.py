import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
import json, os
import sys
from table_instruct.eval.metric.eval_tableinstruct import (
    eval_row_pop_map,
    eval_col_pop_map,
    eval_hitab_ex,
    eval_tabfact_acc,
    eval_col_type_f1,
    eval_ent_link_acc,
    eval_bleu
)
from vllm import LLM, SamplingParams

EOT_TOKEN = "<|EOT|>"


import logging
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

PROMPT_TEMPLATE = """"
table:
{table_info}

Question: {query}
Answer: 
"""

def save_json(result,save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def mklog():
    now = datetime.datetime.now()
    file_path = 'log/' + now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = file_path + '/log.txt'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    # 定义日志输出格式
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(pathname)s-%(lineno)d|: %(message)s')
    # 将格式器应用到处理器上
    file_handler.setFormatter(formatter)
    # 将处理器添加到 logger 实例中
    logger.addHandler(file_handler)
    logger.propagate = False
    return file_path

def build_instruction_prompt_tableinstruct(example):
    #按照tableinstruct的原本的格式来做的,跑出来效果不好
    if "input" in example:
        table_infos = example["input"]
    elif "input_seg" in example:
        table_infos = example["input_seg"]
    else:
        table_infos = ''
    if len(table_infos) > 29897:
        table_infos=table_infos[:29897]+'...'
    query = example["question"]
    instruction=example["instruction"]

    decoder_input_text = f'''<|im_start|>system
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
<|im_end|>
<|im_start|>user
### Instruction:
{instruction}

### Input:
{table_infos}

### Question:
{query}

### Response:<|im_end|>
<|im_start|>assistant
'''

    return decoder_input_text

def build_instruction_prompt(example):
    if "input" in example:
        table_infos = example["input"]
    elif "input_seg" in example:
        table_infos = example["input_seg"]
    else:
        table_infos = ''
    #if len(table_infos) > 29997:
    #    table_infos=table_infos[:29997]+'...'
    #table_infos = example["input"]
    query = example["question"]
    instruction=example["instruction"]

    decoder_input_text = f'''<|im_start|>system
{instruction}
<|im_end|>
<|im_start|>user
table:
{table_infos}

Question: {query}
Answer:<|im_end|>
<|im_start|>assistant
'''

    return decoder_input_text

import json
@torch.inference_mode()
def evaluate(model, tokenizer, output_path, all_data, inference_type, inference_config):
    import tqdm
    logging.info(f'output_path:{output_path}')
    inference_para = json.load(open(inference_config))
    generate_para = inference_para['generate_para']
    if inference_type=='TGI':
        for index, conv in tqdm.tqdm(enumerate(all_data), total=len(all_data)):
            prompt = build_instruction_prompt(conv)
            logging.info(f"prompt for data{index}: {prompt}")
            input_ids=tokenizer.encode(prompt, return_tensors='pt').to(model.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
            generated_ids = model.generate(input_ids, eos_token_id=tokenizer.eos_token_id,
                                           attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, **generate_para)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
            ]
            output_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logging.info(f"output for data{index}: {output_str}")

            output_dict = {
                **conv,
                'predict': output_str
            }
            with open(os.path.join(output_path, 'output.jsonl'), 'a') as f:
                f.write(json.dumps(output_dict) + '\n')
    elif inference_type=='vLLM':
        prompt_batch = []
        for index, conv in tqdm.tqdm(enumerate(all_data), total=len(all_data)):
            # prompt = build_instruction_prompt(conv)
            if "input_seg" in conv:
                table_info = conv["input_seg"]
            elif "input" in conv:
                table_info = conv["input"]
            else:
                table_info = ""
            query = conv["question"]
            instruction = conv["instruction"]
            # table_info = conv["input"]
            prompt_str = PROMPT_TEMPLATE.format(query=query, table_info=table_info)
            msg = [{"role": "system", "content": instruction},{"role": "user", "content": prompt_str}]
            prompt = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            prompt_batch.append(prompt)
        sampling_params = SamplingParams(**generate_para)
        outputs = model.generate(prompt_batch, sampling_params)

        for output, input_data in zip(outputs, all_data):
            input_data["predict"] = output.outputs[0].text
            with open(os.path.join(output_path, 'output.jsonl'), 'a') as f:
                f.write(json.dumps(input_data) + '\n')


@torch.inference_mode()
def evaluate_all(model, tokenizer, json_path, output_path, num_gpus_total, num_gpus_per_model, eval_type, inference_type, inference_config):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #一个测试单元
    for file_name in ['output.jsonl']:
        with open(os.path.join(output_path, file_name), 'w') as f:
            f.write('')        

    if eval_type == 'row_pop':
        all_data=[]
        for file_name in ['part_0.json','part_1.json','part_2.json','part_3.json','part_4.json','part_5.json']:
            sub_path = os.path.join(json_path,file_name)
            sub_data = json.load(open(sub_path))
            all_data += sub_data
    else:
        all_data = json.load(open(json_path))
    #all_data=all_data[:10]#小样本测试
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1
    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            evaluate
        ).remote
    else:
        get_answers_func = evaluate
        
    chunk_size = len(all_data) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(all_data), chunk_size):
        cur_data = all_data[i:i + chunk_size]
        ans_handles.append(get_answers_func(model, tokenizer, output_path, cur_data, inference_type, inference_config))
    if use_ray:
        ray.get(ans_handles)
    result_path = os.path.join(output_path, 'result.json')
    with open(os.path.join(output_path, 'output.jsonl'), 'r') as f:
        dt = [json.loads(line) for line in f]
    result={}
    if eval_type == 'hitab':
        result = eval_hitab_ex(dt)
    elif eval_type in ['fetaqa', 'kvret', 'totto']:
        result = eval_bleu(dt)
    elif eval_type == 'Ent_link':
        result = eval_ent_link_acc(dt)#这个acc特殊
    elif eval_type == 'col_pop':
        result = eval_col_pop_map(dt)
    elif eval_type == 'row_pop':
        result = eval_row_pop_map(dt)#这两个map稍有不同
    elif eval_type in ['col_type', 'rel_extraction']:
        result = eval_col_type_f1(dt)
    elif eval_type in ['tabfact', 'feverous', 'hybridqa', 'wikisql', 'wikitq']:
        result = eval_tabfact_acc(dt)
    save_json(result, result_path)

def evaluate_tableinstruct(model_path, json_path, output_path, num_gpus_total, num_gpus_per_model, dataset_part, inference_type, inference_config):
    inference_para = json.load(open(inference_config))
    load_para = inference_para['load_para']
    if inference_type=='TGI':
        model = AutoModelForCausalLM.from_pretrained(model_path,**load_para).to('cuda')
        model.eval()
    elif inference_type=='vLLM':
        load_para['model']=model_path
        model = LLM(**load_para)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if dataset_part in['in_domain_test','all_test']:
        try:
            # row_pop
            logging.info('Processing row_pop ...')
            print('Processing row_pop ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test', 'row_pop_test')#这个特殊
            output_path_tmp = os.path.join(output_path, 'in_domain_test', 'row_pop_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                            eval_type='row_pop',
                            inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing row_pop error %s', e, exc_info=True)
            print('Processing row_pop error')

        try:
            # col_pop
            logging.info('Processing col_pop ...')
            print('Processing col_pop ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test', 'col_pop_test.json')
            output_path_tmp = os.path.join(output_path, 'in_domain_test', 'col_pop_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='col_pop',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing col_pop error %s', e, exc_info=True)
            print('Processing col_pop error')

        try:
            # col_type
            logging.info('Processing col_type ...')
            print('Processing col_type ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test', 'col_type_test.json')
            output_path_tmp = os.path.join(output_path, 'in_domain_test', 'col_type_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='col_type',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing col_type error %s', e, exc_info=True)
            print('Processing col_type error')

        try:
            #Ent_link
            logging.info('Processing Ent_link ...')
            print('Processing Ent_link ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test', 'ent_link_test.json')
            output_path_tmp = os.path.join(output_path, 'in_domain_test', 'ent_link_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='Ent_link',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing Ent_link error %s', e, exc_info=True)
            print('Processing Ent_link error')

        try:
            #FetaQA
            logging.info('Processing FetaQA ...')
            print('Processing FetaQA ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test', 'fetaqa_test.json')
            output_path_tmp = os.path.join(output_path, 'in_domain_test', 'fetaqa_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model, eval_type='fetaqa',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing FetaQA error %s', e, exc_info=True)
            print('Processing FetaQA error')

        try:
            #Hitab
            logging.info('Processing Hitab ...')
            print('Processing Hitab ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test' ,'hitab_test.json')
            output_path_tmp = os.path.join(output_path, 'in_domain_test' ,'hitab_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model, eval_type='hitab',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing FetaQA error %s', e, exc_info=True)
            print('Processing FetaQA error')

        try:
            # rel_extraction
            logging.info('Processing rel_extraction ...')
            print('Processing rel_extraction ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test', 'rel_extraction_test.json')
            output_path_tmp = os.path.join(output_path, 'in_domain_test', 'rel_extraction_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='rel_extraction',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing rel_extraction error %s', e, exc_info=True)
            print('Processing rel_extraction error')

        try:
            # rel_extraction
            logging.info('Processing tabfact ...')
            print('Processing tabfact ...')
            json_path_tmp = os.path.join(json_path, 'in_domain_test', 'tabfact_test.json')
            output_path_tmp = os.path.join(output_path, 'in_domain_test', 'tabfact_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='tabfact',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing tabfact error %s', e, exc_info=True)
            print('Processing tabfact error')

    if dataset_part in['out_of_domain_test','all_test']:
        try:
            #Feverous
            logging.info('Processing Feverous ...')
            print('Processing Feverous ...')
            json_path_tmp = os.path.join(json_path, 'out_of_domain_test', 'feverous_eval.json')
            output_path_tmp = os.path.join(output_path, 'out_of_domain_test', 'feverous_eval')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='feverous',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing Feverous error %s', e, exc_info=True)
            print('Processing Feverous error')

        try:
            # HybridQA
            logging.info('Processing HybridQA ...')
            print('Processing HybridQA ...')
            json_path_tmp = os.path.join(json_path, 'out_of_domain_test', 'hybridqa_eval.json')
            output_path_tmp = os.path.join(output_path, 'out_of_domain_test', 'hybridqa_eval')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='hybridqa',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing HybridQA error %s', e, exc_info=True)
            print('Processing HybridQA error')

        try:
            # KVRet
            logging.info('Processing KVRet ...')
            print('Processing KVRet ...')
            json_path_tmp = os.path.join(json_path, 'out_of_domain_test', 'kvret_test.json')
            output_path_tmp = os.path.join(output_path, 'out_of_domain_test', 'kvret_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='kvret',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing KVRet error %s', e, exc_info=True)
            print('Processing KVRet error')

        try:
            # ToTTo
            logging.info('Processing ToTTo ...')
            print('Processing ToTTo ...')
            json_path_tmp = os.path.join(json_path, 'out_of_domain_test', 'totto_eval.json')
            output_path_tmp = os.path.join(output_path, 'out_of_domain_test', 'totto_eval')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='totto',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing ToTTo error %s', e, exc_info=True)
            print('Processing ToTTo error')

        try:
            # WikiSQL
            logging.info('Processing WikiSQL ...')
            print('Processing WikiSQL ...')
            json_path_tmp = os.path.join(json_path, 'out_of_domain_test', 'wikisql_test.json')
            output_path_tmp = os.path.join(output_path, 'out_of_domain_test', 'wikisql_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='wikisql',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing WikiSQL error %s', e, exc_info=True)
            print('Processing WikiSQL error')

        try:
            # WikiTQ
            logging.info('Processing WikiTQ ...')
            print('Processing WikiTQ ...')
            json_path_tmp = os.path.join(json_path, 'out_of_domain_test', 'wikitq_test.json')
            output_path_tmp = os.path.join(output_path, 'out_of_domain_test', 'wikitq_test')
            evaluate_all(model, tokenizer, json_path_tmp, output_path_tmp, num_gpus_total, num_gpus_per_model,
                         eval_type='wikitq',
                         inference_type=inference_type, inference_config=inference_config)
        except Exception as e:
            logging.info('Processing WikiTQ error %s', e, exc_info=True)
            print('Processing WikiTQ error')


import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default='table_related_benchmarks/evalset/TableInstruct/eval_data')
    parser.add_argument('--model-path', type=str, default='/data4/sft_output/qwen2.5-ins-1012/checkpoint-2600')
    parser.add_argument('--output-path', type=str, default='table_related_benchmarks/evalset/TableInstruct/eval_data/eval_output-sft')
    parser.add_argument('--num-gpus-total', type=int, default=1)
    parser.add_argument('--num-gpus-per-model', type=int, default=1)
    parser.add_argument('--dataset-part', type=str, default='all_test',
                        choices=['in_domain_test', 'out_of_domain_test', 'all_test'])
    parser.add_argument('--inference-type', type=str, default='vLLM',
                        choices=['TGI', 'vLLM'])
    parser.add_argument('--inference-config', type=str, default='table_related_benchmarks/table_instruct/eval/vLLM_config.json')
    args = parser.parse_args()

    mklog()#这个log其实是有返回值的，但是这里有output_log，暂时不必要
    logging.info(vars(args))
    evaluate_tableinstruct(**vars(args))
    #evaluate_all(**vars(args))
    
if __name__ == '__main__':
    main()
    