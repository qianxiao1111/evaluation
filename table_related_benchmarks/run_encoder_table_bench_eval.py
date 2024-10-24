
import os
import warnings
from inference_encoder import inference_with_encoder, format_encoder_tables, read_df_head, build_encoder_table_part_content
from inference import load_model, load_tokenizer_and_template
from table_bench_eval.run_eval import run_eval, execute_samples_and_save
from table_bench_eval.utils import read_json_file
from typing import List, Dict
import json
import pandas as pd

from table_bench_eval.utils import (
    parse_chart_code_then_exec, 
    parse_code_then_exec, 
    pre_save_table_to_csv,
    parse_final_answer_prediction,
    write_json_to_file,
    execution_eval,
    parse_python_code
)


def format_encoder_inputs(samples: List[Dict]) -> List:
    """
    输入数据格式化函数，按照 generate 的格式要求改造 inputs
    :param samples: 待格式化样例数据
    :param mode: 格式化模式
    """
    
    # 把需要推理的数据拼成 message 形式
    msgs = []
    for sample in samples:
        msg_sys = sample["instruction"]
        table_str = sample["table"]
        table_data = json.loads(table_str)
        pre_save_table_to_csv(table_data)
        # encoder 信息
        df_names = ["table"]
        table_paths = ["table.csv"]
        msg_sys_list = msg_sys.split(table_str)

        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": msg_sys_list[0]},
                    {"type": "text", "text": table_str},
                    *build_encoder_table_part_content(df_names, table_paths),
                    {"type": "text", "text": msg_sys_list[1]},
                ],
            }
        ]
        msgs.append(msg)
    return msgs


def main(args):
    warnings.filterwarnings('ignore')
    inference_output_dir = args.inference_output_dir
    base_model_name = args.model_path
    # 循环四个数据集加载数据进行
    fnames = [x for x in os.listdir(args.eval_dataset_path) if x.endswith('.jsonl')]
    all_samples = []
    n_samples_test = None
    for file_name in fnames:
        print(file_name)
        file_path = os.path.join(args.eval_dataset_path, file_name)
        samples = read_json_file(file_path)
        if n_samples_test:
            samples = samples[:n_samples_test]
        # format messages
        msgs = format_encoder_inputs(samples)
        # inference
        print("Generating eval answers now..")
        model_outputs_text = inference_with_encoder(args, msgs)
        
        print("model_outputs_text", len(model_outputs_text))
        print("Generating answers finished..")
        
        assert len(model_outputs_text) == len(samples)
        for i, output in enumerate(model_outputs_text):
            samples[i]["raw_generation"] = output
        
        save_path = os.path.join(inference_output_dir, base_model_name.split('/')[-1]+'_infer_'+file_name.split('.')[0]+'.jsonl')
        with open(save_path, 'w') as f:
            for item in samples:
                f.write(json.dumps(item)+'\n')
        all_samples.extend(samples)

    # get execuate results
    all_samples = execute_samples_and_save(all_samples, inference_output_dir, base_model_name)
    # eval and save results
    run_eval(all_samples, inference_output_dir, base_model_name)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Table bench evaluation")

    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )

    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Temperature setting"
    )

    parser.add_argument(
        "--model_path", type=str, help="Path to the model", default="/data4/sft_output/qwen2.5-7b-base-0923/checkpoint-2000"
    )

    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="table_related_benchmarks/evalset/TableBench",
        help="Test Set Path",
    )

    parser.add_argument(
        "--inference_output_dir",
        type=str,
        default="table_related_benchmarks/evalset/TableBench/eval_results",
        help="Max iteration for llm to run each code correction task",
    )

    parser.add_argument(
        "--model_type",
        choices=["base_model", "chat_model"],
        default="chat_model",
        help="Base model or Chat model",
    )

    parser.add_argument(
        "--max_model_len", type=int, default=15000, help="Max model length"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of output tokens",
    )

    parser.add_argument(
        "--template",
        type=str,
        choices=[None, "llama3", "baichuan", "chatglm"],
        default=None,
        help="The template must be specified if not present in the config file",
    )

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)