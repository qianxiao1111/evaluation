import re
from reject_eval.prompt import (
    eval_system,
    eval_instruction,
    output_content_classify_instruct,
    output_content_classify_system
)
from utils import filter_code
from reject_eval.eval_metrics import evaluation
from utils import load_json, save_json
import os

def contains_independent_no(text):
    # \b 表示单词边界, \s* 表示0个或多个空白字符，包括空格、制表符和换行符
    pattern = r'\bno\b\s*'
    match = re.search(pattern, text, re.IGNORECASE)
    return match is not None

def format_inputs(test_datas: list[dict]) -> list[list[dict]]:
    """Format inputs to the required messages"""
    # 把需要推理的数据拼成 message 形式
    format_message_datas = []
    for idx, test_dt in enumerate(test_datas):
        query = test_dt["query"]
        df_info_str = test_dt["df_info"]

        format_instruction = eval_instruction.format(df_info=df_info_str, input=query)
        format_system = eval_system.format(df_info=df_info_str, input=query)

        messages = [
            {"role": "system", "content": format_system},
            {"role": "user", "content": format_instruction}
        ]
        format_message_datas.append(messages)
    
    return format_message_datas

def format_llm_outputs(model_outputs: list[dict]) -> list[list[dict]]:
    format_message_datas = []
    for sample in model_outputs:
        sentence = sample["output_text"]
        format_instruction = output_content_classify_instruct.format(input=sentence)
        messages = [
            {"role": "system", "content": output_content_classify_system},
            {"role": "user", "content": format_instruction}
        ]
        format_message_datas.append(messages)

    return format_message_datas

def eval_outputs(model_outputs: list[dict], test_file_path: str, save_path: str = "") -> None:
    """Calculate the reject evaluation metric based
    on model outputs for binary classification
    """
    test_datas = load_json(test_file_path)
    # 提取模型输出list
    output_texts = [i["output_text"] for i in model_outputs]
    processed_data = []
    for idx, test_dt in enumerate(test_datas):
        llm_output = output_texts[idx]

        test_dt["llm_output"] = llm_output
        code, pure_code = filter_code(llm_output)
        if pure_code == "" or contains_independent_no(pure_code):
            test_dt["is_reject"] = True
        else:
            test_dt["is_reject"] = False


        # # 解析输出判断结果
        # if any(keyword.lower() in llm_output.lower() for keyword in ["positive"]):
            
        # elif any(keyword.lower() in llm_output.lower() for keyword in ["negative"]):
        #     test_dt["is_reject"] = True

            # print("解析错误")
        processed_data.append(test_dt)
    
    # 保存路径
    parent_path = os.path.dirname(test_file_path)
    if not save_path:
        save_path = os.path.join(parent_path, 'llm_output_data.json')
    ground_truth_path = os.path.join(parent_path, 'ground_truth.json')
    ground_truth_datas = load_json(ground_truth_path)
    for i in range(len(ground_truth_datas)):
        processed_data[i]["true_result"] = ground_truth_datas[i]["is_reject"]
        # processed_data[i][""]

    save_json(save_path, processed_data)
    print(f"评估每条数据的模型输出及结果保存路径：{save_path}")
    evaluation(ground_truth_path, save_path)

