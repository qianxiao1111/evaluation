# 看下 error_answer 里有哪些数据集，从 wk02上同步下来
import json

with open("gen/error/error_answer.json", "r") as f:
    samples = json.load(f)

ls_tables = []
for sample in samples:
    ls_tables.extend([t.split("/")[1] for t in sample["tables"]])

print(set(ls_tables))
