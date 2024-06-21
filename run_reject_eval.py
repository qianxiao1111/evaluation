import argparse
import time
from reject_eval.run_eval import format_inputs, load_json, eval_outputs
from inference import load_model, load_tokenizer_and_template, generate_outputs

def main(args):
    temperature = args.temperature
    model_path = args.model_path
    max_new_tokens = args.max_new_tokens
    max_model_len = args.max_model_len
    test_path = args.test_path
    template = args.template
    gpus_num = args.gpus_num

    # 加载 model 和 tokenizer
    llm_model = load_model(model_path, max_model_len, gpus_num)
    tokenizer = load_tokenizer_and_template(model_path, template)

    # 推理参数
    generate_args = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    # 推理&评估
    test_datas = load_json(test_path)
    format_message_datas = format_inputs(test_datas)
    model_outputs = generate_outputs(format_message_datas, llm_model, tokenizer, generate_args)
    eval_outputs(model_outputs, test_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval reject")
    parser.add_argument('--gpus_num', type=int, default=1, help='the number of GPUs you want to use.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature setting')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of output new tokens')
    parser.add_argument('--max_model_len', type=int, default=8192, help='Max model length')
    parser.add_argument('--template', type=str, choices=[None, 'llama3', 'baichuan', 'chatglm'], default=None, help='The template must be specified if not present in the config file')
    parser.add_argument('--test_path', type=str, default="evalset/reject_test/test_query.json", help='Test File Path')

    args = parser.parse_args()
    main(args)


# example /home/dev/weights/CodeQwen1.5-7B-Chat /data0/pretrained-models/checkpoints/qwen2/checkpoint-1200 
"""
python run_reject_eval.py \
    --model_path /data1/sft_models/checkpoints/v0619/qwen2-7b-instruct  \
    --temperature 0 \
    --max_model_len 16384 \
    --max_new_tokens 1024
"""
