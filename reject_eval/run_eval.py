from reject_eval.prompt import eval_system, eval_instruction
from reject_eval.eval_metrics import evaluation
from util import load_json, save_json
import os


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
        # 解析输出判断结果
        if "yes" in llm_output.lower():
            test_dt["is_reject"] = False
        elif "no" in llm_output.lower():
            test_dt["is_reject"] = True
        else:
            pass
            # print("解析错误")
        processed_data.append(test_dt)
    
    # 保存路径
    parent_path = os.path.dirname(test_file_path)
    if not save_path:
        save_path = os.path.join(parent_path, 'llm_output_data.json')
    ground_truth_path = os.path.join(parent_path, 'ground_truth.json')

    save_json(save_path, processed_data)
    print(f"评估每条数据的模型输出及结果保存路径：{save_path}")
    evaluation(ground_truth_path, save_path)
