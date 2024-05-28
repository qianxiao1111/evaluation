# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/28 14:52
@Auth ： zhaliangyu
@File ：llms.py
@IDE ：PyCharm
"""
from langchain_openai import ChatOpenAI

llm_gen = ChatOpenAI(
    temperature=0.01,
    max_tokens=1024,
    verbose=True,
    openai_api_base="http://localhost:8080/v1",
    openai_api_key="none",
    model_name="deepseek-coder-6.7b-instruct",
)

llm_for_eval = ChatOpenAI(
    temperature=0.01,
    max_tokens=2048,
    verbose=True,
    openai_api_key="",
    model_name="gpt-4o-2024-05-13",
    # model_kwargs={"stop": stop},
)

llm_judge = ChatOpenAI(
    temperature=0.7,
    max_tokens=1024,
    verbose=True,
    openai_api_key="",
    model_name="gpt-4o-2024-05-13",
        # model_kwargs={"stop": stop},
)