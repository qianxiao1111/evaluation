from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json
import argparse
import transformers


def get_examples_messages(data_path, tokenizer):
    test_dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train",
    )
    examples = []
    messages_list = []

    systems = test_dataset["system"]
    instructions = test_dataset["instruction"]
    inputs = test_dataset["input"]
    outputs = test_dataset["output"]
    historys = test_dataset["history"]
    sources = test_dataset["source"]
    for i in tqdm(range(len(instructions))):
        history = historys[i]
        input = instructions[i] + "\n" + inputs[i]
        output = outputs[i]
        system = systems[i]
        messages = []
        if len(system):
            messages.append({"role": "system", "content": system})
        if len(history):
            for his in history:
                messages.append({"role": "user", "content": his[0]})
                messages.append({"role": "assistant", "content": his[1]})
        messages.append({"role": "user", "content": input})
        messages_list.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        examples.append(
            {
                "instruction": instructions[i],
                "input": inputs[i],
                "output": output,
                "system": system,
                "history": history,
                "source": sources[i],
            }
        )
        # messages.append({"role": "assistant", "content": output})
    print(len(messages_list))
    print(len(examples))
    return examples, messages_list


def generate_result(examples, messages_list, args):
    llm_args = {
        "model": args.model_path,
        "gpu_memory_utilization": 0.95,
        "trust_remote_code": True,
        "tensor_parallel_size": args.gpus_num,
        "dtype": "half",
        "max_model_len": 8192,
        "enforce_eager": True,
    }
    llm = LLM(**llm_args)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_p=0.95,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    outputs = llm.generate(messages_list, sampling_params=sampling_params)
    generated_examples = []
    for i, output in enumerate(tqdm(outputs)):
        output = output.outputs[0].text
        example = examples[i]
        example["output_tablegpt"] = output
        generated_examples.append(example)
    with open(args.save_path, "w") as f:
        json.dump(generated_examples, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model name or path",
        default="/data4/sft_output/qwen2-base-0804/checkpoint-1800/",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="model name or path",
        default="/data3/yss/sft_datas/0817/sft_data_merge_v17_quality_test.jsonl",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="model name or path",
        default="output/result_evaldataset_final.json",
    )
    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )

    args = parser.parse_args()
    transformers.set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    examples, messages_list = get_examples_messages(args.data_path, tokenizer)
    generate_result(examples, messages_list, args)
