import json
import re
import warnings
import pandas as pd
import os
import argparse
import shutil
import copy
from pathlib import Path
from joblib import Parallel, delayed
from io import StringIO
import pandas as pd

from inference_encoder import inference_with_encoder, format_encoder_tables, read_df_head, build_encoder_table_part_content
from utils import (
    get_tool,
    filter_code,
    timeout,
    TimeoutException,
    load_json,
)

CODE_PREFIX = """import matplotlib.pyplot as plt
from mplfonts import use_font
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# Fixing Chinese font issues
use_font("Noto Serif CJK SC")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False\n"""

def eval_outputs_parallel(
    llm_output: str,
    test_data: str,
    args,
) -> dict:
    df_paths = test_data["table_paths"]
    df_names = test_data["df_names"]
    query = test_data["query"]
    table_paths = test_data["table_paths"]
    df = [pd.read_csv(path, low_memory=False) for path in df_paths]

    if args.slim:
        # tool = get_tool(df, df_names)
        tool = get_tool(df) 
        instruction = test_data["message"]
    else:
        tool = get_tool(df, df_names)
        instruction = test_data["instruction"]
        table_info = test_data["table_info"]
        df_info_simple_str = test_data["df_info_simple_str"]
        instruction = instruction.replace(table_info, df_info_simple_str)

    code, _ = filter_code(llm_output)
    # cot = filter_cot(llm_output)
    eval_result_sample = {}
    # 运行超时代码，认为都是异常代码， 在tool.run()过程中，可能会print出额外的内容，不影响执行
    try:
        # 如果生成的代码为空（解析不到代码）， 也认为是llm没有理解observe内容或instruct， 输出为Code Error
        if not code:
            observe = "Code Error: output empty code.."
        elif 'df.explode("Candidate")' in code:
            raise ValueError(f"df.explode error")
        else:
            with timeout(15):  # 设置超时时间为15秒
                pure_code = CODE_PREFIX + code
                # print("pure code:", pure_code)
                observe = tool.run(pure_code)  # 需要监控超时的代码块
                # observe = execute_with_timeout(pure_code, 15, tool)
                if isinstance(observe, pd.DataFrame):
                    observe = observe.head().to_markdown(index=False)
                else:
                    observe = str(observe)
    except TimeoutException as e:
        observe = f"Timeout Error: code running time exceed 15s.."
    except SystemExit as e:
        observe = f"SystemExit Error: {str(e)}"
    except Exception as e:
        observe = f"Unexpected Error: {str(e)}"

    eval_result_sample["code"] = code
    eval_result_sample["llm_output"] = llm_output
    eval_result_sample["observe"] = observe
    eval_result_sample["flag"] = execution_eval(observe)
    eval_result_sample["query"] = query
    eval_result_sample["table_paths"] = table_paths
    eval_result_sample["instruction"] = instruction

    return eval_result_sample

def execution_eval(observe: str) -> bool:
    """
    Test whether the code generated by eval_llm can be executed.
    :param output: output code of llm generation
    :return: True or False
    """
    # 只要执行结果中不出现error 或者 exception， 就认为代码可执行
    pattern = re.compile(r"error|exception", re.IGNORECASE)
    try:
        res = not pattern.search(observe)
    except:
        res = True
    return res

def extract_df_info(df: pd.DataFrame):
    sio = StringIO()
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.info(buf=sio, memory_usage=False)

    return sio.getvalue()

def build_single_slim_messages(test_dt):
    query = test_dt["query"]
    instruction = test_dt["instruction"]
    table_info = test_dt["table_info"]
    df_info_simple_str = test_dt["df_info_simple_str"]
    table_paths = test_dt["table_paths"]
    df_names = test_dt["df_names"]

    messages = [{
        "role": "system",
        "content": "You are 数海闻涛, an expert Python data analyst developed by 浙江大学计算机创新技术研究院 (Institute of Computer Innovation of Zhejiang University, or ZJUICI). Your job is to help user analyze datasets by writing Python code. Each markdown codeblock you write will be executed in an IPython environment, and you will receive the execution output. You should provide results analysis based on the execution output.\nFor politically sensitive questions, security and privacy issues, or other non-data analyze questions, you will refuse to answer.\n\nRemember:\n- Comprehend the user's requirements carefully & to the letter.\n- If additional information is needed, feel free to ask the user.\n- Give a brief description for what you plan to do & write Python code.\n- You can use `read_df(uri: str) -> pd.DataFrame` function to read different file formats into DataFrame.\n- When creating charts, prefer using `seaborn`.\n- If error occurred, try to fix it.\n- Response in the same language as the user.\n- Today is 2024-09-26"
    }]

    table_info_messages = []
    for idx, table_path in enumerate(table_paths):
        df_name = f"df{idx + 1}"
        file_name = os.path.basename(table_path)
        df_head_str, df = read_df_head(table_path, 3)
        content_msg = build_encoder_table_part_content([df_name], [table_path])
        table_info_messages.extend(copy.deepcopy(
            [
                {
                    "role": "user",
                    "content": f"文件名称: '{file_name}'"
                },
                {
                    "role": "assistant",
                    "content": f"我已经收到您的数据文件，我需要查看文件内容以对数据集有一个初步的了解。首先我会读取数据到 `{df_name}` 变量中，并通过 `{df_name}.info` 查看 NaN 情况和数据类型。\n\n```python\n# Load the data into a DataFrame\n{df_name} = read_df('{file_name}')\n\n# Remove leading and trailing whitespaces in column names\n{df_name}.columns = {df_name}.columns.str.strip()\n\n# Remove rows and columns that contain only empty values\n{df_name} = {df_name}.dropna(how='all').dropna(axis=1, how='all')\n\n# Get the basic information of the dataset\n{df_name}.info(memory_usage=False)\n```"
                },
                {
                    "role": "system",
                    "content": f"```pycon\n{extract_df_info(df)}\n```"
                },
                {
                    "role": "assistant",
                    "content": f"接下来我将用 `{df_name}.head(3)` 来查看数据集的前 3 行。\n\n```python\n# Show the first 3 rows to understand the structure\n{df_name}.head(3)\n```"
                },
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"```pycon\n{df_head_str}\n```"},
                        *content_msg,
                    ],
                },
                {
                    "role": "assistant",
                    "content": "我已经了解了数据集 {file_name} 的基本信息。请问我可以帮您做些什么？"
                }
            ])
        )

    messages.extend(table_info_messages)
    messages.append({"role": "user", "content": query})

    return messages

def build_tableqa_messages_from_csv_file(test_dt):
    query = test_dt["query"]
    instruction = test_dt["instruction"]
    table_info = test_dt["table_info"]
    df_info_simple_str = test_dt["df_info_simple_str"]
    table_paths = test_dt["table_paths"]
    df_names = test_dt["df_names"]

    # instruction 切分重新拼接 messages
    instruction_list = instruction.split(table_info)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": instruction_list[0]},
            {"type": "text", "text": df_info_simple_str},
            *build_encoder_table_part_content(df_names, table_paths),
            {"type": "text", "text": instruction_list[1]},
        ],
    })
    return messages

def format_inputs(test_datas: list[dict],args) -> list[list[dict]]:
    """Format inputs to the required messages"""
    # 把需要推理的数据拼成 message 形式
    format_message_datas = []
    for idx, test_dt in enumerate(test_datas):
        if args.slim:
            messages = build_single_slim_messages(test_dt)
        else:
            messages = build_tableqa_messages_from_csv_file(test_dt)
        if messages:
            format_message_datas.append(messages)

    return format_message_datas


def main(args):
    warnings.filterwarnings('ignore')
    eval_dataset_path = args.eval_dataset_path
    eval_results_save_path = args.eval_results_save_path
    # load eval dataset
    eval_dataset_path = args.eval_dataset_path
    test_datas = load_json(eval_dataset_path)
    format_message_datas = format_inputs(test_datas, args)
    print("Generating eval answers now..")
    # inference
    model_outputs_text = inference_with_encoder(args, format_message_datas)
    print("model_outputs_text", len(model_outputs_text))
    # eval
    eval_answers = Parallel(n_jobs=48)(
        delayed(eval_outputs_parallel)(model_outputs_text[i], test_datas[i],args)
        for i in range(len(test_datas))
    )
    # calculate  execute rate
    execute_passed = 0
    total_len = len(eval_answers)
    for eval_answer in eval_answers:
        execute_passed += int(eval_answer["flag"])
    print(f"Sample length: {total_len}. ")
    print(
        f"Execute Passed: {execute_passed}." f"\tExecute pass-rate is:",
        round(execute_passed / total_len, 3),
    )
    # save eval result
    with open(eval_results_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_answers, f, ensure_ascii=False)


if __name__ == "__main__":
    # 确定images目录是否存在和写权限
    output_dir = Path(__file__).parent / "images"
    if os.path.exists(output_dir):
        if not os.access(output_dir, os.W_OK):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            os.chmod(output_dir, 0o777)
            print("not write permission, makedir:", output_dir)
        else:
            print(f"{output_dir} exists!")
    else:
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o777)
        print("makedir:", output_dir)
    parser = argparse.ArgumentParser(description="eval tableqa python code")
    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Temperature setting"
    )

    parser.add_argument(
        "--template",
        type=str,
        choices=[None, "llama3", "baichuan", "chatglm"],
        default=None,
        help="The template must be specified if not present in the config file",
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
        "--slim",
        action="store_true",
        help="slim data format",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of output tokens",
    )
    parser.add_argument("--max_model_len", type=int, default=10000, help="Cutoff length")
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="table_related_benchmarks/evalset/table_qa_execuate_test/test_datas.json",
        help="Test Set Path",
    )

    parser.add_argument(
        "--eval_results_save_path",
        type=str,
        default="output/result_table_qa.json",
        help="Max iteration for llm to run each code correction task",
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
