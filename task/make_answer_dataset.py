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
from task.utils.executor import executor_on_db, read_table_from_db
from task.utils.load import load_df, load_json, load_openai_llm
from task.utils.pytool import PythonAstREPLTool
import warnings

warnings.filterwarnings(action="ignore")


def get_foreign_keys(table_name: str, conn: sqlite3.connect) -> List[Tuple]:
    """查询单个表的外键关系"""
    cursor = conn.cursor()
    query = f"PRAGMA foreign_key_list('{table_name}')"
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return [(row[2], row[3], row[4]) for row in results]  # 获取父表名和对应的列名


def get_foreign_key_relation(
    table_names: List[str], conn: sqlite3.connect
) -> List[str]:
    """遍历查询所有表的外键关系"""
    all_foreign_keys = []
    for table_name in table_names:
        foreign_keys = get_foreign_keys(table_name, conn)
        formatted_relations = [
            f"df{int(table_names.index(table_name) + 1)}.{fk[1]} = df{int(table_names.index(fk[0]) + 1)}.{fk[2]}"
            for fk in foreign_keys
            if (table_name in table_names) & (fk[0] in table_names)
        ]
        all_foreign_keys.extend(formatted_relations)
    return all_foreign_keys


def get_table_data(
    table_names: List[str], conn: sqlite3.connect
) -> Dict[str, pd.DataFrame]:
    result_dict = {}
    for table_name in table_names:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn)
        result_dict[table_name] = df
    return result_dict


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


CODE_PREFIX = """import matplotlib.pyplot as plt
from mplfonts import use_font
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# Fixing Chinese font issues
use_font("Noto Serif CJK SC")\n"""
path_to_csv = "datasets/csv_lower/{}/{}.csv"
path_to_query = "datasets/evalset/querys.json"
path_to_tables = "datasets/evalset/y_tables.json"
benches = {
    "BIRD_dev": "datasets/BIRD_dev/dev_databases",
    "SPIDER_dev": "datasets/spider/database",
}
key_query = {
    "BIRD_dev": "SQL",
    "SPIDER_dev": "query",
}

samples = load_json(path_to_query)
y_tables = load_json(path_to_tables)
llm = load_openai_llm(
    "http://localhost:8083", model_name="deepseek-coder-6.7b-instruct", max_tokens=8192
)  # 使用一个弱模型 deepseek-6.7b，尽量生成错误答案
gen = PyGenChain.from_llm(llm)
assert len(samples) == len(y_tables), "length not fetch"
results = []
answers = []  # 趁这个处理过程，把执行结果也写到一个独立file里
for i in tqdm(range(len(samples))):
    sample = samples[i]
    y_table = y_tables[i]
    question = sample["question"]
    path_to_sqlite = os.path.join(
        benches[sample["from"]], sample["db_id"], sample["db_id"] + ".sqlite"
    )
    answer = executor_on_db(
        query=sample[key_query[sample["from"]]], path_to_db=path_to_sqlite
    )
    answers.append(str(answer))

    # gen python code
    try:
        db_name = "{}-{}".format(sample["from"], sample["db_id"])
        table_infos, df_locals = get_table_info(y_table, path_to_sqlite)
        plan, code, ori = gen.predict(table_infos=table_infos, query=question)
        ast = PythonAstREPLTool(locals=df_locals)
        exec_code = CODE_PREFIX + code
        flag, exec = ast(exec_code)
        results.append(
            {
                "id": "Retriever-{}".format(i),
                "tables": [path_to_csv.format(db_name, table) for table in y_table],
                "table_infos": table_infos,
                "query": question,
                "thought_cot": str(plan),
                "python_code": str(exec_code),
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
    "datasets/code_and_exec/answers_BIRD_SPIDER.json", "w", encoding="utf-8"
) as f:
    json.dump(answers, f, ensure_ascii=False)
with open(
    "datasets/code_and_exec/code_and_exec_BIRD_SPIDER.json", "w", encoding="utf-8"
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
    "datasets/code_and_exec/error_answer_BIRD_SPIDER.json", "w", encoding="utf-8"
) as f:
    json.dump(errors, f, ensure_ascii=False)
print("错误集", len(errors))
