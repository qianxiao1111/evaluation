import os
import sys

sys.path.append(".")

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from task.gen import PyGenChain
from task.utils.load import load_df, load_json
from task.utils.pytool import PythonAstREPLTool
import warnings

warnings.filterwarnings(action="ignore")

# llm = ChatOpenAI(
#     temperature=0.01,
#     model="glm-4",
#     openai_api_key="b80d5056b182aedcbef71f6a3a25e7c5.HTjtJPZpI7s8zZrZ",
#     openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
# )
llm = ChatOpenAI(
    temperature=0.01,
    max_tokens=2048,
    verbose=True,
    openai_api_key="",
    model_name="gpt-4o-2024-05-13",
    # model_kwargs={"stop": stop},
)

samples = load_json("evalset/error_answer_yss.json")
errors = []
py_gen_chain = PyGenChain.from_llm(llm)

for sample in tqdm(samples):
    try:
        question = sample["query"]
        path_to_csv = "evalset/csv/" + sample["tables"][0].split("./")[1]
        df = load_df(path_to_csv)
        ast = PythonAstREPLTool(locals={"df": df, "pd": pd, "np": np})
        table_head = df.head(3).to_markdown()
        table_infos = "`df`:\n{}".format(table_head)
        cot, code, ori = py_gen_chain.predict(table_infos=table_infos, query=question)
        flag, exec = ast(code)
        sample["exec_boll_gpt4"] = flag
        sample["exec_result_gpt4"] = str(exec)
        sample["code_gpt4"] = str(code)
        sample["cot_gpt4"] = str(cot)
        errors.append(sample)

        # 只能每次写一遍了
        with open("evalset/error_answer_yss_gpt.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False)
    except Exception as e:
        print(e)
