import argparse
import warnings
import os
import pandas as pd
from utils import filter_code, load_json, save_json
from reject_eval.run_eval import contains_independent_no, load_json
from reject_eval.eval_metrics import evaluation
from inference_encoder import inference_with_encoder, format_encoder_tables, read_df_head, build_encoder_table_part_content
from reject_eval.prompt import (eval_instruction, eval_system,
                                output_content_classify_instruct,
                                output_content_classify_system)


def format_encoder_inputs(test_datas: list[dict]) -> list[list[dict]]:
    """Format inputs to the required messages"""
    # 把需要推理的数据拼成 message 形式
    format_message_datas = []
    for idx, test_dt in enumerate(test_datas):
        query = test_dt["query"]
        df_info_str = test_dt["df_info"]
        table_paths = test_dt["table_paths"]
        table_paths = [os.path.join("table_related_benchmarks", table_path) for table_path in table_paths]
        df_names = test_dt["df_names"]
        
        # encoder 信息
        # tables, encoder_tables_info = format_encoder_tables(df_names, table_paths)
        content_msg = build_encoder_table_part_content(df_names, table_paths)
        # 只有单表数据
        if len(table_paths) != 1:
            raise ValueError("多表情况")

        # df_info_str = df_info_str + f"\n/*\n{encoder_tables_info[0].strip()}\n*/"
        format_instruction = eval_instruction.format(df_info=df_info_str, input=query)
        format_instruction_list = format_instruction.split(df_info_str)

        format_system = eval_system
        messages = [
            {
                "role": "system", 
                "content": format_system
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": format_instruction_list[0]},
                    {"type": "text", "text": df_info_str},
                    *content_msg,
                    {"type": "text", "text": format_instruction_list[1]},
                ],
            }
        ]
        format_message_datas.append(messages)

    return format_message_datas

def eval_encoder_outputs(
    model_outputs: list[dict], test_file_path: str, save_path: str = ""
) -> None:
    """Calculate the reject evaluation metric based
    on model outputs for binary classification
    """
    test_datas = load_json(test_file_path)
    # 提取模型输出list
    output_texts = model_outputs
    processed_data = []
    for idx, test_dt in enumerate(test_datas):
        llm_output = output_texts[idx]

        test_dt["llm_output"] = llm_output
        code, pure_code = filter_code(llm_output)
        if pure_code == "" or contains_independent_no(pure_code):
            test_dt["is_reject"] = True
        else:
            test_dt["is_reject"] = False
        
        processed_data.append(test_dt)

    # 保存路径
    parent_path = os.path.dirname(test_file_path)
    if not save_path:
        save_path = os.path.join(parent_path, "llm_output_data.json")
    ground_truth_path = os.path.join(parent_path, "ground_truth.json")
    ground_truth_datas = load_json(ground_truth_path)
    for i in range(len(ground_truth_datas)):
        processed_data[i]["true_result"] = ground_truth_datas[i]["is_reject"]
        # processed_data[i][""]
        if processed_data[i]["true_result"] == processed_data[i]["is_reject"]:
            processed_data[i]["flag"] = True
        else:
            processed_data[i]["flag"] = False

    save_json(save_path, processed_data)
    print(f"评估每条数据的模型输出及结果保存路径：{save_path}")
    evaluation(ground_truth_path, save_path)

def main(args):
    warnings.filterwarnings('ignore')
    test_path = args.test_path
    # load eval datasets
    test_datas = load_json(test_path)
    # 推理
    format_message_datas = format_encoder_inputs(test_datas)
    print("Generating eval answers now..")
    model_outputs_text = inference_with_encoder(args, format_message_datas)
    print("model_outputs_text", len(model_outputs_text))
    print("Generating answers finished..")
    # 评估
    eval_encoder_outputs(model_outputs_text, test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval reject")
    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Temperature setting"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--model_type",
        choices=["base_model", "chat_model"],
        default="chat_model",
        help="Base model or Chat model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of output new tokens",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=15000, help="Max model length"
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=[None, "llama3", "baichuan", "chatglm"],
        default=None,
        help="The template must be specified if not present in the config file",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="table_related_benchmarks/evalset/reject_test/test_query.json",
        help="Test File Path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/result_reject.json",
        help="LLM output samples save path",
    )

    args = parser.parse_args()
    main(args)

