import argparse
from util import load_json
from recall_eval.run_eval import (
    format_inputs,
    parser_list,
    eval_outputs,
    save_result,
)
from inference import load_model, load_tokenizer_and_template, generate_outputs


def main(args):
    # init model
    llm_model = load_model(args.model_path, args.max_model_len, args.gpus_num)
    tokenizer = load_tokenizer_and_template(args.model_path, args.template)
    generate_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "model_type": args.model_type,
    }
    report = dict()
    preds = dict()
    samples = load_json(args.test_path)
    if args.num is not None:
        samples = samples[: args.num]

    # sql test
    msgs_gen_sql = format_inputs(samples, "gen_sql")
    resp_gen_sql = generate_outputs(msgs_gen_sql, llm_model, tokenizer, generate_args)
    code_gen_sql = parser_list(resp_gen_sql, "gen_sql")
    msgs_ext_sql = format_inputs(code_gen_sql, "extract_sql")
    resp_ext_sql = generate_outputs(msgs_ext_sql, llm_model, tokenizer, generate_args)
    pred_ext_sql = parser_list(resp_ext_sql, "extract_sql")
    report["sql"] = eval_outputs(pred_ext_sql, samples)
    preds["resp_gen_sql"] = resp_gen_sql
    preds["resp_ext_sql"] = resp_ext_sql

    # python test
    msgs_gen_py = format_inputs(samples, "gen_python")
    resp_gen_py = generate_outputs(msgs_gen_py, llm_model, tokenizer, generate_args)
    code_gen_py = parser_list(resp_gen_py, "gen_python")
    msgs_ext_py = format_inputs(code_gen_py, "extract_python")
    resp_ext_py = generate_outputs(msgs_ext_py, llm_model, tokenizer, generate_args)
    pred_ext_py = parser_list(resp_ext_py, "extract_python")
    report["python"] = eval_outputs(pred_ext_py, samples)
    preds["resp_gen_py"] = resp_gen_py
    preds["resp_ext_py"] = resp_ext_py

    # save result
    print(report)
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
        '--model_type', choices=['base_model', 'chat_model'], default="chat_model", help='Base model or Chat model'
    )
    parser.add_argument(
        '--gpus_num', type=int, default=1, help='the number of GPUs you want to use.'
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature setting"
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
        default="evalset/recall_set.json",
        help="Test File Path",
    )
    parser.add_argument("--num", type=int, default=None, help="number of lines to eval")
    args = parser.parse_args()
    main(args)


# example /home/dev/weights/CodeQwen1.5-7B-Chat /data0/pretrained-models/checkpoints/qwen2/checkpoint-1200
"""
python run_recall_eval.py \
    --model_path /home/qyhuang/weights/CodeQwen1.5-7B-Chat \
    --temperature 0.01 \
    --max_model_len  8192 \
    --max_new_tokens 1024 \
    --test_path evalset/retrieval_test/recall_set.json 
"""
