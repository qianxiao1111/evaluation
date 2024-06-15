# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/28 14:52
@Auth ： zhaliangyu
@File ：llms.py
@IDE ：PyCharm
"""
from langchain_openai import ChatOpenAI

# llm for the eval_method `llm_eval`
llm_judge = ChatOpenAI(
    temperature=0.7,
    max_tokens=2048,
    verbose=True,
    openai_api_key="",
    model_name="gpt-4o-2024-05-13",
        # model_kwargs={"stop": stop},
)