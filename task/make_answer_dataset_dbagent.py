# 跑一遍 eval_retriever/data/querys.json 的问题，生成 python 与 执行结果
# 判断对错，错误的直接计入新的 error_answer，参照 yss 格式，加一个 answer 字段（yss版本 缺少 answer 需要补充）
# 添加标记，合并成新的dataset
import os
import sys

sys.path.append(".")
import json
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
from task.gen import PyGenChain
from task.utils.load import load_df, load_json, load_openai_llm
from task.utils.pytool import PythonAstREPLTool
import warnings

warnings.filterwarnings(action="ignore")


def get_table_info(table_names: List[str], path_to_db: str, topk: int = 1):
    table_info = ""
    df_locals = {}
    conn = sqlite3.connect(path_to_db)
    df_dict = get_table_data(table_names, conn)
    if len(table_names) == 1:
        table_info += "df:\n{}\n".format(
            df_dict[table_names[0]].head(topk).to_markdown()
        )
        df_locals["df"] = df_dict[table_names[0]]
    else:
        for table_name, df in df_dict.items():
            table_info += "df{}:\n{}\n".format(
                int(table_names.index(table_name) + 1), df.head(3).to_markdown()
            )
            df_locals["df{}".format(int(table_names.index(table_name) + 1))] = df
        fk_info = get_foreign_key_relation(table_names, conn)
        table_info += "Foreign keys:\n{}\n".format(",".join(fk_info))
    conn.close()
    df_locals["pd"] = pd
    df_locals["np"] = np
    return table_info, df_locals


path_to_tables = "datasets/da-dev/da-dev-tables/{}"
path_to_labels = "datasets/da-dev/da-dev-labels.jsonl"
path_to_questions = "datasets/da-dev/da-dev-questions.jsonl"

# init
samples = load_json(path_to_questions)
labels = load_json(path_to_labels)
llm = load_openai_llm(
    "http://localhost:8083", model_name="deepseek-coder-6.7b-instruct", max_tokens=8192
)  # 使用一个弱模型 deepseek-6.7b，尽量生成错误答案
gen = PyGenChain.from_llm(llm)
assert len(samples) == len(labels), "length not fetch"
results = []
answers = []  # 趁这个处理过程，把执行结果也写到一个独立file里
for i in tqdm(range(len(samples))):
    sample = samples[i]
    label = [label for label in labels if label["id"] == sample["id"]][0]
    question = sample["question"]
    answer = label["common_answers"]

    # gen python code
    try:
        df = load_df(path_to_tables.format(sample["file_name"]))
        table_infos = "df:\n{}\n".format(df.head(3).to_markdown())
        df_locals = {}
        df_locals["pd"] = pd
        df_locals["np"] = np
        df_locals["df"] = df
        plan, code, ori = gen.predict(table_infos=table_infos, query=question)
        ast = PythonAstREPLTool(locals=df_locals)
        flag, exec = ast(code)
        results.append(
            {
                "id": sample["id"],
                "tables": [path_to_tables.format(sample["file_name"])],
                "table_infos": table_infos,
                "query": question,
                "thought_cot": str(plan),
                "python_code": str(code),
                "exec_bool": flag,
                "exec_result": str(exec),
                "answer": str(answer),
                "llm_result": str(ori),
            }
        )
    except Exception as e:
        print(e)
        pass


with open(
    "datasets/code_and_exec/code_and_exec_db_agent.json", "w", encoding="utf-8"
) as f:
    json.dump(results, f, ensure_ascii=False)
print("总集合", len(results))

errors = [
    result
    for result in results
    if (result["exec_bool"] is False)
    & (result["exec_result"] != "NameError: name 'pd' is not defined")
    & (result["exec_result"] != "NameError: name 'np' is not defined")
]
with open(
    "datasets/code_and_exec/error_answer_db_agent.json", "w", encoding="utf-8"
) as f:
    json.dump(errors, f, ensure_ascii=False)
print("错误集", len(errors))
