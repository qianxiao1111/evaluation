
import os
import pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch


def load_tokenizer_and_template(model_name_or_path, template=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if tokenizer.chat_template is None:
        if template is not None:
            chatml_jinja_path = pathlib.Path(os.path.dirname(os.path.abspath(
                __file__))) / f"templates/template_{template}.jinja"
            assert chatml_jinja_path.exists()
            with open(chatml_jinja_path, "r") as f:
                tokenizer.chat_template = f.read()
        else:
            raise ValueError("chat_template is not found in the config file, please provide the template parameter.")
    return tokenizer


def load_model(model_name_or_path, max_model_len=None, gpus_num=1):
    llm_args = {
        "model": model_name_or_path,
        "gpu_memory_utilization": 0.8,
        "trust_remote_code": True,
        "tensor_parallel_size": gpus_num,
        "dtype": "half",
    }

    if max_model_len:
        llm_args["max_model_len"] = max_model_len
        
    # Create an LLM.
    llm = LLM(**llm_args)
    return llm

def generate_outputs(messages_batch, llm_model, tokenizer, generate_args):
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    messages_batch = [messages]

    generate_args = {
        "max_new_tokens": 1024,
        "do_sample": True or False,
        "temperature": 0-1,
        ""
    }
    """
    sampling_params = SamplingParams(**generate_args)
    
    prompt_batch = []
    for messages in messages_batch:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_batch.append(prompt)
    
    outputs = llm_model.generate(prompt_batch, sampling_params)

    outputs_batch = []
    for output in outputs:
        prompt_output = output.prompt
        generated_text = output.outputs[0].text
        outputs_batch.append({
            "input_prompt": prompt_output,
            "output_text": generated_text
        })
    
    return outputs_batch


if __name__ == "__main__":
    model_name_or_path = "/home/dev/weights/CodeQwen1.5-7B-Chat"
    template = "baichuan"
    generate_args = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 1024
    }
    max_model_len = 16384
    tokenizer = load_tokenizer_and_template(model_name_or_path, template)
    llm_model = load_model(model_name_or_path, max_model_len)

    messages = [
        # {"role": "system", "content": "You are a python assistant."},
        {"role": "user", "content": "list dict 根据某个key排序"}
    ]
    messages_batch = [messages] * 1000
    outputs_batch = generate_outputs(messages_batch, llm_model, tokenizer, generate_args)
    print(outputs_batch)
