import json
import argparse
from text2sql.src.evaluation import evaluation_main
from text2sql.src.evaluation_ves import evaluation_ves_main

from text2sql.src.gpt_request import generate_main

def main(args):
    if args.eval_data_name == "bird":
        args.db_root_path = "table_related_benchmarks/text2sql/data/bird_data/dev_databases"
        args.eval_data_path = "table_related_benchmarks/text2sql/data/bird_data/dev.json"
        args.ground_truth_path = "table_related_benchmarks/text2sql/data/bird_data/dev.sql"
        args.mode = "dev"
        args.use_knowledge = "True"
        
    if args.eval_data_name == "spider":
        args.db_root_path = "table_related_benchmarks/text2sql/data/spider_data/test_database"
        args.eval_data_path = "table_related_benchmarks/text2sql/data/spider_data/test.json"
        args.ground_truth_path = "table_related_benchmarks/text2sql/data/spider_data/test_gold.sql"
        args.mode = "test"
        args.use_knowledge = "False"
    
    if args.is_use_knowledge:
        args.use_knowledge = "True"
    eval_datas = json.load(open(args.eval_data_path, 'r'))

    # '''for debug'''
    # datas = []
    # simple_num = 0
    # moderate_num = 0
    # challenging_num = 0
    # filter_num = 3
    # for i,data in enumerate(eval_datas):
    #     if simple_num== moderate_num==challenging_num== filter_num:
    #         break
    #     if eval(f"{data['difficulty']}_num") >= filter_num:
    #         continue
        
    #     if data['difficulty'] == 'simple':
    #         datas.append(data)
    #         simple_num += 1

    #     if data['difficulty'] == 'moderate':
    #         datas.append(data)
    #         moderate_num += 1

    #     if data['difficulty'] == 'challenging':
    #         datas.append(data)
    #         challenging_num += 1
    # import copy
    # eval_datas = copy.deepcopy(datas)
    # '''for debug'''
    predicted_sql_path = generate_main(eval_datas, args)
    # predicted_sql_path = "table_related_benchmarks/text2sql/output/predict_dev.json"
    evaluation_main(args, eval_datas, predicted_sql_path)
 

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_data_name', type=str, choices=["bird", "spider"], default="bird")

    args_parser.add_argument('--is_use_knowledge', default=False, action="store_true")
    args_parser.add_argument('--data_output_path', type=str, default="table_related_benchmarks/text2sql/output")
    args_parser.add_argument('--chain_of_thought', type=str, default="False")
    args_parser.add_argument('--model_path', type=str, required=True)
    args_parser.add_argument('--gpus_num', type=int, default=1)
    args_parser.add_argument('--num_cpus', type=int, default=4)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    
    args = args_parser.parse_args()
    main(args)


    # CUDA_VISIBLE_DEVICES=6 python table_related_benchmarks/run_text2sql_eval.py --model_path /data4/sft_output/qwen2.5-7b-ins-0929/checkpoint-3200 --eval_data_name spider
    
