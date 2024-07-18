"""
@Time ： 2024/6/15 09:52
@Auth ： zhaliangyu
@File ：run_code_correction_eval.py
@IDE ：PyCharm
"""

# todo 筛选eval-set， 剔除数据过大的样本， 并新增一批样本
import json
from evaluate_code_correction.run_eval import format_inputs, eval_outputs, run_eval
from inference import load_model, load_tokenizer_and_template, generate_outputs


def check_eval_dataset_keys(test_datas):
    pass


def get_infer_kwargs(args) -> dict:
    """llm_inference kwargs"""
    temperature = args.temperature if args.temperature else 1.0
    max_new_tokens = args.max_new_tokens if args.max_new_tokens else 1024
    model_type = args.model_type if args.model_type else "chat_model"

    kwargs = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "model_type": model_type,
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
    model_outputs = generate_outputs(
        format_message_datas, llm_model, tokenizer, model_kwargs
    )
    print("Generating answers finished..")

    # this is the step to generate all the eval_answers first
    eval_answers = eval_outputs(
        model_outputs,
        eval_dataset_path,
        test_csv_file_path=test_csv_file_path,
        lan_type="Python",
    )
    print("Eval answers construct complete..")
    with open(eval_results_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_answers, f, ensure_ascii=False)
    # this is the step to get eval_pass_rate
    run_eval(eval_result_path=eval_results_save_path, llm_for_judge=llm_for_judge)


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
        "--test_csv_file_path",
        type=str,
        help="Path to the test csv files",
        default="./",
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
        default="evalset/code_correction_test/correction_set.json",
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
    main(args)

    """example:
    python run_code_correction_eval.py --model_path /home/qyhuang/weights/deepseek-coder-6.7b-instruct --test_csv_file_path "./"


    python run_code_correction_eval.py \
    --model_path /home/dev/weights/CodeQwen1.5-7B-Chat \
    --test_csv_file_path . \
    --eval_dataset_path evalset/code_correction_test/correction_set.json \
    --eval_results_save_path evalset/code_correction_test/results_qwen7b.json 

    python run_code_correction_eval.py \
    --model_path /home/dev/weights/Qwen1.5-72B-Chat-GPTQ-Int4 \
    --test_csv_file_path . \
    --eval_dataset_path evalset/code_correction_test/correction_set.json \
    --eval_results_save_path evalset/code_correction_test/results_qwen72b.json \
    --gpus_num 4
    """
