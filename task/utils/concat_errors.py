import sys
import json
import warnings

sys.path.append(".")
from task.utils.load import load_json, load_df

warnings.filterwarnings(action="ignore")

# gpt4制作的错误答案，不太可能让 glm4生成正确的
zg = load_json("datasets/code_and_exec/error_answer_BIRD_SPIDER.json")
print("SPIDER & BIRD:", len(zg))

da = load_json("datasets/code_and_exec/error_answer_db_agent.json")
print("db-agent total:", len(da))

yss = load_json("datasets/code_and_exec/error_answer_yss_glm.json")
print("yss total:", len(yss))
yss = [
    sample for sample in yss if sample["exec_boll_glm4"]
]  # 筛选 glm4 执行正确的是正确答案，因为我也没有 gpt4 api
print("yss clean:", len(yss))
samples = []
for sample in zg:
    samples.append(
        {
            "table_paths": sample["tables"],
            "query": sample["query"],
            "cot": sample["thought_cot"],
            "code": sample["python_code"],
            "observation": sample["exec_result"],
            "true_result": sample["answer"],
            "table_infos": sample["table_infos"],
        }
    )

for sample in da:
    samples.append(
        {
            "table_paths": sample["tables"],
            "query": sample["query"],
            "cot": sample["thought_cot"],
            "code": sample["python_code"],
            "observation": sample["exec_result"],
            "true_result": sample["answer"],
            "table_infos": sample["table_infos"],
        }
    )


for sample in yss:
    path = sample["tables"][0].replace("./", "datasets/csv/")
    samples.append(
        {
            "table_paths": [path],
            "query": sample["query"],
            "cot": sample["thought_cot"],
            "code": sample["python_code"],
            "observation": sample["exec_result"],
            "true_result": sample["exec_result_glm4"],
            "table_info": "df:\n{}".format(load_df(path).head(3).to_markdown()),
        }
    )

with open("datasets/evalset/correction_set.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False)
print("correction_set:", len(samples))
