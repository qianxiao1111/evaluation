"""
- [ ] 1. 针对 SPIDER+BIRD 完成用 glm4(chain) 的代码生成
- [ ] 2. 测试 sql_eval 项目中的 eval 检测方案（df标准化、df比对、df子集校验）
- [ ] 3. 晚上 python 代码生成的功能，实现 多表的代码生成 与 单元测试 
- [ ] 4. 完善 recorrection 数据集（重新生成带多表的）
- [ ] 5. 优化 recorrection 的实现逻辑并 prompt
"""

import os
from task.gen import PyGenChain
from task.utils.load import load_json, load_df, load_openai_llm
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.01,
    model="glm-4",
    openai_api_key="b80d5056b182aedcbef71f6a3a25e7c5.HTjtJPZpI7s8zZrZ",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)
llm_ds = load_openai_llm(
    "http://localhost:8083", model_name="deepseek-coder-6.7b-instruct"
)
gen = PyGenChain.from_llm(llm=llm)
gen_ds = PyGenChain.from_llm(llm=llm_ds)

# dataset
samples = load_json("/mnt/temp/querys.json")
y_tables = load_json("/mnt/temp/y_tables.json")
y_columns = load_json("/mnt/temp/y_columns.json")
sample = samples[0]
y_table = y_tables[0]
db_id = "{}-{}".format(sample["from"], sample["db_id"])
df = load_df(os.path.join("/mnt/temp/datasets/csv", db_id, f"{y_table[0]}.csv"))

table_infos = df.head(3).to_markdown()
query = sample["question"]

plan, resp, ori = gen.predict(table_infos=table_infos, query=query)
plan_ds, resp_ds, ori_ds = gen_ds.predict(table_infos=table_infos, query=query)

print(samples[0])

# execute
import numpy as np
import pandas as pd
from task.utils.pytool import PythonAstREPLTool

# 最好再做个前处理去掉 read_df
ast = PythonAstREPLTool(locals={"df": df, "pd": pd, "np": np})
flag, exec = ast(resp.replace("output.csv", "output_glm.csv"))
flag_ds, exec_ds = ast(resp_ds.replace("output.csv", "output_ds.csv"))

# sql_eval helper
from task.sql_eval_func.helper import normalize_table, compare_df, subset_df

df_glm = load_df("output_glm.csv")
df_ds = load_df("output_ds.csv")
df_glm = normalize_table(df_glm)
df_ds = normalize_table(df_ds)
compare_df(df_glm, df_ds)
subset_df(df_glm, df_ds)


from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
