import sys
import json

sys.path.append(".")
from task.utils.load import load_json

# gpt4制作的错误答案，不太可能让 glm4生成正确的
zg = load_json("datasets/evalset/error_answer_spider_bird_new.json")
print("SPIDER & BIRD:", len(zg))
yss = load_json("datasets/evalset/error_answer_yss_glm_new.json")
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
        }
    )


for sample in yss:
    samples.append(
        {
            "table_paths": [
                tb.replace("./", "datasets/csv/") for tb in sample["tables"]
            ],
            "query": sample["query"],
            "cot": sample["thought_cot"],
            "code": sample["python_code"],
            "observation": sample["exec_result"],
            "true_result": sample["exec_result_glm4"],
        }
    )

with open("datasets/evalset/correction_set.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False)
print("total:", len(samples))
