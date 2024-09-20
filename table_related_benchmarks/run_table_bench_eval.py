
import os 
from inference import load_model, load_tokenizer_and_template
from table_bench_eval.run_eval import model_infer_and_save, run_eval, execute_samples_and_save

def main(args):
    llm_model = load_model(args.model_path, args.max_model_len, args.gpus_num)
    tokenizer = load_tokenizer_and_template(args.model_path, args.template)
    inference_output_dir = args.inference_output_dir
    base_model_name = args.model_path
    generate_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "model_type": args.model_type,
        "top_p": 0.95,
        "n": 1
    }
    # inference for output 
    all_samples = model_infer_and_save(args.eval_dataset_path, llm_model, tokenizer, generate_args, inference_output_dir, base_model_name)
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
        "--model_path", type=str, help="Path to the model", default="/data4/sft_output/qwen2-base-table-0916/checkpoint-2400"
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
        "--max_model_len", type=int, default=16384, help="Max model length"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
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


# Example
"""
python table_related_benchmarks/run_table_bench_eval.py
"""