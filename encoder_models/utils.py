import os
import transformers
from typing import Dict
from encoder_models.encoder1.config import INSERT_EMBS_TOKEN, INSERT_EMBS_TOKEN_ID
import torch

def find_correct_case_file_name(path, name):
    ls = os.listdir(path)
    ls = [x.split('.')[0] for x in ls]
    for gt in ls:
        if gt.lower() == name.lower():
            return gt
    # 找为子串的
    for gt in ls:
        if name.lower() in gt.lower():
            return gt
    raise ValueError(f'path {path}, name "{name}" not found')

def build_plain_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

def build_instruction_prompt(question, content):

    decoder_input_text = f'''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction: 
You have access to a number of SQL tables. Given a user question about the data, write the SQL query to answer it.
Notes: Don't assume you have access to any tables other than the ones provided. You MUST only write the SQL query, nothing else, in the format of a single string.
You MUST only write the SQL query, nothing else, in the format of a single string, like 'SELECT count(*) FROM head WHERE val > 114514'. You MUST NOT include any explanation or context in the answer.
Only the provided tables can be used in the SQL query.
### Table Information: 
{content}
### Question: 
{question}
### Response:
'''

    return decoder_input_text


def tokenize_insert(prompt: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    '''
    Tokenizes the input prompt by inserting a separator token between each chunk of text.

    Args:
        prompt (str): The input prompt to be tokenized. It contains one or more instances of the INSERT_EMBS_TOKEN.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer object used for tokenization.

    Returns:
        torch.Tensor: The tokenized input prompt as a tensor of input IDs. You need to move to the correct device before using it.

    '''
    prompt_chunks = [tokenizer(e, padding="longest", max_length=tokenizer.model_max_length, truncation=True).input_ids for e in prompt.split(INSERT_EMBS_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id: # tokenizer会在每次encode前面都加一个bos_token_id
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [INSERT_EMBS_TOKEN_ID] * (offset + 1)):  # insert separator 返回的是 [chunk1, [sep] * (offset + 1), chunk2, [sep] * (offset + 2), ...]，然后用offset统一减掉一个
        input_ids.extend(x[offset:])
    return torch.tensor(input_ids, dtype=torch.long)

def ray_work(func, data, num_gpus, num_gpus_per_worker, devices):
    import ray
    NUM_GPUS = num_gpus
    os.environ['CUDA_VISIBLE_DEVICES']=devices
    NUM_GPUS_PER_WORKER = num_gpus_per_worker
    NUM_PROCESSES = int(NUM_GPUS // NUM_GPUS_PER_WORKER)
    print(f'NUM_GPUS: {NUM_GPUS}, NUM_GPUS_PER_WORKER: {NUM_GPUS_PER_WORKER}, NUM_PROCESSES: {NUM_PROCESSES}')

    ray.shutdown()
    ray.init()
    CHUNK_SIZE = len(data) // NUM_PROCESSES + 1
    get_answers_func = ray.remote(num_gpus=NUM_GPUS_PER_WORKER)(func,).remote
    cur_data = [data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(NUM_PROCESSES)]
    print(len(cur_data))
    futures = [get_answers_func(tt_data) for tt_data in cur_data]
    ret = ray.get(futures)
    ray.shutdown()
    return ret

def build_instruction_qwen(prompt, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    decoder_input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return decoder_input_text