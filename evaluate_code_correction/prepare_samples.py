# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/25 16:16
@Auth ： zhaliangyu
@File ：prepare_samples.py
@IDE ：PyCharm
"""
import re
import datetime
import pandas as pd
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from prompt import PROMPT_PYTHON_GENERATE_NORMAL, PROMPT_PYTHON_GENERATE_VISUAL
# from utils import llm_gen
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool


def extract_before_python_block(text):
    # Find the index where the ```python block starts
    python_block_index = text.find("Python Code")

    # If the ```python block is found, return the text before it
    if python_block_index != -1:
        return text[:python_block_index]
    else:
        # If no ```python block is found, return the original text
        return text

CODE_PREFIX = """import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.max_rows', 6)
"""

llm_gen = ChatOpenAI(
    temperature=0.01,
    max_tokens=1024,
    verbose=True,
    openai_api_base="http://localhost:8080/v1",
    openai_api_key="none",
    model_name="deepseek-coder-6.7b-instruct",
)


test_data = "./test_data/small_fund_table.csv"
df = pd.read_csv(test_data)
table_infos = df.head(3).to_markdown()
test_queries = pd.read_csv("./test_data/test_queries.csv")
queries = test_queries["queries"].values
samples = []

current_time = datetime.datetime.now().strftime('%Y-%m-%d:%H')
prompt = ChatPromptTemplate.from_messages(
    [("user", PROMPT_PYTHON_GENERATE_NORMAL)]
)

chain = LLMChain(
    llm = llm_gen,
    prompt=prompt
)
regex = r"```python\s(.*?)```"

samples = []
for query in queries:
    sample = {}
    sample["query"] = query
    sample["table_infos"] = table_infos
    sample["table_paths"] = test_data
    sample["true_result"] = ""
    res = chain.invoke(
        input = {"input":query,  "df_head": table_infos, "current_time": current_time}
    )
    python_tool = PythonAstREPLTool()
    locals = {"df": df}
    python_tool.locals = locals
    python_tool.globals = python_tool.locals
    gen_content = res["text"]
    action_match = re.search(regex, gen_content, re.DOTALL)
    tool_prefix = CODE_PREFIX
    action_input = action_match.group(1)
    action_input = action_input.strip(" ")
    action_input = action_input.strip('"')
    code = action_input.strip(" ")
    code = tool_prefix + code
    code_res = python_tool.run(code)
    sample["code"] = code
    sample["observation"] = code_res
    sample["cot"] = extract_before_python_block(gen_content)
    pattern = re.compile(r"error|exception", re.IGNORECASE)
    try:
        r = not pattern.search(code_res)
    except:
        r = True
    if not r:
        sample["execute_result"] = ""
        print("code:", code)
    else:
        sample["execute_result"] = code_res
    samples.append(sample)
    # if there is no print func in the last line of code, add to it
    print("Sample:", sample)
import json

with open("./test_data/test_samples.json", "w") as f:
    json.dump(samples, f, ensure_ascii=False)