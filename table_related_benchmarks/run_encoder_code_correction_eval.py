
import json
import os
from vllm import LLM
from vllm.sampling_params import SamplingParams
from joblib import Parallel, delayed
import warnings
import copy
from tqdm import tqdm
from inference_encoder import inference_with_encoder, format_encoder_tables, read_df_head, build_encoder_table_part_content
from evaluate_code_correction.run_eval import eval_outputs_parallel
from evaluate_code_correction.prompt import RECTIFY_PROMPT_PYTHON_INSTRUCTION
from utils import load_json

def format_encoder_inputs(test_datas: list[dict], lan_type: str = "Python") -> list[list[dict]]:
    """
    Format inputs with prompts and input variances
    :param test_datas: loaded eval samples
    :param lan_type: Code type, support [`Python`] now
    :return
    """
    # 把需要推理的数据拼成 message 形式
    format_message_datas = []
    for idx, sample in tqdm(enumerate(test_datas)):
        queries = sample["query"]
        table_paths = sample["table_paths"]
        table_paths = [os.path.join("table_related_benchmarks", table_path) for table_path in table_paths]

        # df_names
        if len(table_paths) == 1:
            df_names = ["df"]
        else:
            df_names = [f"df{i+1}" for i in range(len(table_paths))]
        # encoder 信息
        # tables, encoder_tables_info = format_encoder_tables(df_names, table_paths)
        # content_msgs = build_encoder_table_part_content(df_names, table_paths)

        # 原来的df_head信息和encoder的一起拼接
        all_tables_msgs = []
        for idx, table_path in enumerate(table_paths):
            df_head_str, df = read_df_head(table_path, 3, "md")
            df_name = df_names[idx]
            # table_path = table_paths[idx]
            content_msg = build_encoder_table_part_content([df_name], [table_path])
            text_content = f"""/*\n"{df_names[idx]}.head(3).to_markdown()" as follows:\n{df_head_str}\n*/"""
            all_tables_msgs.append(copy.deepcopy({"type": "text", "text": text_content}))
            all_tables_msgs.extend(copy.deepcopy(content_msg))
        # print(all_tables_msgs)
        # 设定当前时间
        current_time = "2024-09-26"
        output = (
            "Thought: " + sample["cot"] + "\n```python\n" + sample["code"] + "\n```"
        )
        observes = sample["observation"]

        format_instruction = RECTIFY_PROMPT_PYTHON_INSTRUCTION.format(
            table_infos="<TABLE_CONTENT>",
            query=queries,
            observe=observes,
            current_time=current_time,
            output=output,
        )
        format_instruction_list = format_instruction.split("<TABLE_CONTENT>")
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": format_instruction_list[0]},
                    *all_tables_msgs,
                    {"type": "text", "text": format_instruction_list[1]},
                ],
            }
        ]
        format_message_datas.append(messages)

    return format_message_datas


def main(args):
    """main function to run evaluation"""
    warnings.filterwarnings('ignore')
    eval_dataset_path = args.eval_dataset_path
    eval_results_save_path = args.eval_results_save_path
    # load eval dataset
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        test_datas = json.load(f)
    format_message_datas = format_encoder_inputs(copy.deepcopy(test_datas))
    if args.run_llm_eval:
        from evaluate_code_correction.llms import llm_judge
        llm_for_judge = llm_judge
    else:
        llm_for_judge = None
    
    # inference generating answers
    print("Generating eval answers now..")
    model_outputs_text = inference_with_encoder(args, format_message_datas)
    print("model_outputs_text", len(model_outputs_text))
    print("Generating answers finished..")

    # this is the step to generate all the eval_answers first
    # test_datas = load_json(eval_dataset_path)
    eval_answers = Parallel(n_jobs=48)(
        delayed(eval_outputs_parallel)(
            model_outputs_text[i], test_datas[i]
        )
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

    with open(eval_results_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_answers, f, ensure_ascii=False)

    # this is the step to get eval_pass_rate
    # run_eval(eval_result_path=eval_results_save_path, llm_for_judge=llm_for_judge)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="eval code_correction")
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
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of output tokens",
    )
    parser.add_argument("--max_model_len", type=int, default=8192, help="Cutoff length")
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="table_related_benchmarks/evalset/code_correction_test/correction_set.json",
        help="Test Set Path",
    )
    parser.add_argument(
        "--eval_results_save_path",
        type=str,
        default="output/result_code_correction.json",
        help="Eval results save path",
    )
    parser.add_argument(
        "--run_llm_eval",
        type=bool,
        default=False,
        help="Whether use another llm to judge the eval-results, if set to `True`, modify the `evaluate_code_correction/llms.py` configs",
    )
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
