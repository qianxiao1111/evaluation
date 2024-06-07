import json
import pandas as pd
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceTextGenInference
from langchain.schema.language_model import BaseLanguageModel


def load_df(path_to_file: str, nrows=None) -> pd.DataFrame:
    if path_to_file.endswith(".csv"):
        return pd.read_csv(path_to_file, nrows=nrows)
    elif path_to_file.endswith(".xlsx"):
        return pd.read_excel(path_to_file, nrows=nrows)
    else:
        raise Exception("Invlid path.")


def read_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_llm(url: str) -> BaseLanguageModel:
    llm = HuggingFaceTextGenInference(
        inference_server_url=url,
        max_new_tokens=1024,
        temperature=0.01,
        repetition_penalty=1.03,
        seed=42,
        stop_sequences=[
            "\nNew_Question:",
            "\nQuestion:",
        ],
    )
    return llm


def load_qwen(infer_url) -> BaseLanguageModel:
    llm = ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="{}/v1".format(infer_url),
        model_name="qwen1.5-14b-awq",
        max_tokens=8192,
        temperature=0.01,
        # model_kwargs={"stop": ["<|im_end|>"]},
    )
    return llm


def load_gpt() -> BaseLanguageModel:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0.01,
        openai_api_key="",
        max_tokens=8192,
    )
    return llm


def load_gpt4() -> BaseLanguageModel:
    llm = ChatOpenAI(
        model_name="gpt-4-turbo",
        temperature=0.01,
        openai_api_key="",
    )
    return llm


def load_json(filepath: str):
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]


def load_hf_llm(url: str) -> BaseLanguageModel:
    llm = HuggingFaceTextGenInference(
        inference_server_url=url,
        max_new_tokens=1024,
        temperature=0.01,
        repetition_penalty=1.03,
        seed=42,
        stop_sequences=[
            "\nNew_Question:",
            "\nQuestion:",
        ],
    )
    return llm


def load_openai_llm(
    url: str,
    model_name="qwen1.5-14b-awq",
    max_tokens=None,
    temperature=0.01,
) -> BaseLanguageModel:
    llm = ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="{}/v1".format(url),
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return llm
