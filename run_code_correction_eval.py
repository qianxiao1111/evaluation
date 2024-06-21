"""
@Time ： 2024/6/15 09:52
@Auth ： zhaliangyu
@File ：run_code_correction_eval.py
@IDE ：PyCharm
"""

import json
import argparse
from tqdm import tqdm
from evaluate_code_correction.run_eval import (
    format_inputs,
    eval_outputs,
    run_eval
)

from inference import load_model, load_tokenizer_and_template, generate_outputs

def check_eval_dataset_keys(test_datas):
    pass


def get_infer_kwargs(args) -> dict:
    """llm_inference kwargs"""
    temperature = args.temperature if args.temperature else 1.0
    max_new_tokens = args.max_new_tokens if args.max_new_tokens else 1024

    kwargs = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    return kwargs

def main(args):
    """main function to run evaluation"""

    eval_dataset_path = args.eval_dataset_path
    eval_results_save_path = args.eval_results_save_path
    model_path = args.model_path
    test_csv_file_path = args.test_csv_file_path
    max_model_len = args.max_model_len
    template = args.template
    gpus_num = args.gpus_num
    model_kwargs = get_infer_kwargs(args)
    print("Load model...")
    llm_model = load_model(model_path, max_model_len, gpus_num)
    tokenizer = load_tokenizer_and_template(model_path, template)
    print("Model load success..")
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        test_datas = json.load(f)
    format_message_datas = format_inputs(test_datas)
    if args.run_llm_eval:
        from evaluate_code_correction.llms import llm_judge
        llm_for_judge = llm_judge
    else:
        llm_for_judge = None
    # generating answers
    print("Generating eval answers now..")
    model_outputs = generate_outputs(format_message_datas,
                                         llm_model,
                                         tokenizer,
                                         model_kwargs)
    with open("results_model_outputs.json", "w") as f:
        json.dump(model_outputs, f, ensure_ascii=False)
    # with open("results_model_outputs.json", "r", encoding="utf-8") as f:
    #     model_outputs = json.load(f)
    print("Generating answers finished..")

    # this is the step to generate all the eval_answers first
    eval_answers = eval_outputs(model_outputs,
                                eval_dataset_path,
                                test_csv_file_path=test_csv_file_path,
                                lan_type="Python")
    print("Eval answers construct complete..")
    with open(eval_results_save_path, "w") as f:
        json.dump(eval_answers, f, ensure_ascii=False)
    # this is the step to get eval_pass_rate
    run_eval(eval_result_path=eval_results_save_path,
             test_csv_file_path=test_csv_file_path,
             llm_for_judge=llm_for_judge)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="eval code_correction")
    parser.add_argument('--gpus_num', type=int, default=1, help='the number of GPUs you want to use.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature setting')
    parser.add_argument('--template', type=str, choices=[None, 'llama3', 'baichuan', 'chatglm'], default=None, help='The template must be specified if not present in the config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--test_csv_file_path', type=str, required=True, help='Path to the test csv files' ,default="./")
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of output tokens')
    parser.add_argument('--max_model_len', type=int, default=8192, help='Cutoff length')
    parser.add_argument('--eval_dataset_path', type=str, default="evalset/code_correction_test/correction_set.json", help='Test Set Path')
    parser.add_argument('--eval_results_save_path', type=str, default="evalset/code_correction_test/results.json", help='Max iteration for llm to run each code correction task')
    parser.add_argument('--run_llm_eval', type=bool, default=False, help='Whether use another llm to judge the eval-results, if set to `True`, modify the `evaluate_code_correction/llms.py` configs')
    args = parser.parse_args()
    main(args)

    """example:
    python run_code_correction_eval.py --model_path /home/qyhuang/weights/deepseek-coder-6.7b-instruct --test_csv_file_path "./"
    """

