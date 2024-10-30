import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import transformers
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.utils import extract_generation_code, languge_settings
from vllm import LLM, SamplingParams

data_abs_dir = Path(__file__).parent / "data"


def build_deepseekcoder_instruction(languge: str, question: str):
    return """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(
        languge.lower(), question.strip()
    )


def create_dir(output_dir):
    if os.path.exists(output_dir):
        if not os.access(output_dir, os.W_OK):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            os.chmod(output_dir, 0o777)
            print("not write permission, makedir:", output_dir)
        else:
            print(f"{output_dir} exists!")
    else:
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o777)
        print("makedir:", output_dir)


def get_client_res(messages, example, output_key, open_ai_key=False):
    try:
        if open_ai_key:
            from openai import AzureOpenAI, OpenAI
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                print("环境变量 OPENAI_API_KEY 未设置")
                api_key = "default_value"

            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-07-01-preview",
                azure_endpoint="https://zju-tablegpt.openai.azure.com/",
            )
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                # model="gpt-4o-mini",
                messages=messages,
                top_p=0.95,
                temperature=0,
                max_tokens=1024,
                timeout=40,
            )
        else:
            # Set OpenAI's API key and API base to use vLLM's API server.
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8080/v1"

            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            chat_response = client.chat.completions.create(
                model="qwen2-7b-sft",
                messages=messages,
                top_p=0.3,
                temperature=0.1,
                max_tokens=1024,
            )
        example[output_key] = chat_response.choices[0].message.content
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        example[output_key] = None
    example["input"] = messages
    return example



def generate_main(args):
    model_name_or_path = args.model_path
    lang = args.language
    temp_dir = args.temp_dir
    create_dir(temp_dir)
    # os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
    if not args.api:
        print("model", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(
            "load tokenizer {} from {} over.".format(
                tokenizer.__class__, model_name_or_path
            )
        )
        llm_args = {
            "model": model_name_or_path,
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

    examples = [json.loads(x) for x in open(problem_file) if x.strip()]
    print("Read {} examples for evaluation over.".format(len(examples)))
    messages_list = []
    for example in tqdm(examples, desc="Generating"):
        prompt = build_deepseekcoder_instruction(
            languge_settings[lang]["full_name"], example["prompt"]
        )
        message = [{"role": "user", "content": prompt}]
        if args.api:
            messages_list.append(message)
        else:
            messages_list.append(
                tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
            )
    if args.api:
        from joblib import Parallel, delayed
        examples_ = Parallel(n_jobs=24)(
            delayed(get_client_res)(inp, examples[i], "output",open_ai_key=True)
            for i, inp in enumerate(tqdm(messages_list))
        )

        # 请求错误的重新请求
        examples = []
        for example in examples_:
            if example["output"] == None:
                example = get_client_res(
                    example["input"], example, "output", open_ai_key=True
                )
            del example["input"]
            examples.append(example)

        generated_examples = []
        for example in examples:
            example = extract_generation_code(example, lang_code=lang)
            generated_examples.append(example)
    else:
        outputs = llm.generate(messages_list, sampling_params=sampling_params)
        generated_examples = []
        for i, output in enumerate(tqdm(outputs)):
            output = output.outputs[0].text
            example = examples[i]
            example["output"] = output
            example = extract_generation_code(example, lang_code=lang)
            generated_examples.append(example)

    print("Generate all over!!!")
    # os.makedirs(args.save_dir, exist_ok=True)
    create_dir(args.save_dir)
    saved_path = os.path.join(args.save_dir, "results_humaneval.json")
    with open(saved_path, "w", encoding="utf-8") as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + "\n")
        print(
            "Save {} processed examples into {} over!".format(
                len(generated_examples), saved_path
            )
        )

    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang,
        out_path=saved_path,
    )
    print(lang, result, model_name_or_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model name or path",
        default="/data4/sft_output/qwen2-instruct-0709/checkpoint-1400",
    )
    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="output path of your generation",
        default="output",
    )
    parser.add_argument("--api", action="store_true", help="infer api type")
    parser.add_argument("--language", type=str, help="langauge", default="python")
    parser.add_argument(
        "--temp_dir", type=str, help="temp dir for evaluation", default="output/tmp"
    )
    parser.add_argument("--seed", type=int, help="seed", default=42)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.set_seed(args.seed)
    generate_main(args)
