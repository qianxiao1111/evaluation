
import os
import pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch


def load_tokenizer_and_template(model_name_or_path, template=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    if tokenizer.chat_template is None:
        if template is not None:
            chatml_jinja_path = pathlib.Path(os.path.dirname(os.path.abspath(
                __file__))) / f"templates/template_{template}.jinja"
            assert chatml_jinja_path.exists()
            with open(chatml_jinja_path, "r") as f:
                tokenizer.chat_template = f.read()
        else:
            pass
            # raise ValueError("chat_template is not found in the config file, please provide the template parameter.")
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
    model_type = generate_args.pop("model_type", "chat_model")

    sampling_params = SamplingParams(**generate_args)
    
    prompt_batch = []
    for messages in messages_batch:
        # 如果是basemodel， 直接拼接prompt内容后输入到模型
        if model_type == "base_model":
            messages_content = [msg["content"] for msg in messages]
            prompt = "\n".join(messages_content)
        # 如果是chat—model 则拼接chat-template后输入
        else:
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
