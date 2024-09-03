import argparse

from inference import generate_outputs, load_model, load_tokenizer_and_template
from recall_eval.run_eval import (eval_outputs,
                                                           format_inputs,
                                                           make_pred,
                                                           parser_list,
                                                           pprint_format,
                                                           save_result)
from utils import load_json


def main(args):
    # init model
    llm_model = load_model(args.model_path, args.max_model_len, args.gpus_num)
    tokenizer = load_tokenizer_and_template(args.model_path, args.template)
    generate_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "model_type": args.model_type,
    }

    samples = load_json(args.test_path)
    if args.num is not None:
        samples = samples[: args.num]

    #  test
    msgs = format_inputs(samples)
    resp = generate_outputs(msgs, llm_model, tokenizer, generate_args)
    pred = parser_list(resp)
    report = eval_outputs(pred, samples)
    preds = make_pred(samples, pred)
    # save result
    pprint_format(report)
    save_result(preds, report, args.test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval recall")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default="/home/dev/weights/CodeQwen1.5-7B-Chat",
        help="Path to the model",
    )
    parser.add_argument(
        "--model_type",
        choices=["base_model", "chat_model"],
        default="chat_model",
        help="Base model or Chat model",
    )
    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Temperature setting"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of output new tokens",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192, help="Max model length"
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
        default="table_related_benchmarks/evalset/retrieval_test/recall_set.json",
        help="Test File Path",
    )
    parser.add_argument("--num", type=int, default=None, help="number of lines to eval")
    args = parser.parse_args()
    main(args)


"""
CUDA_VISIBLE_DEVICES=6 python table_related_benchmarks/run_recall_eval.py --model_path /data4/sft_output/qwen2-base-0817
"""