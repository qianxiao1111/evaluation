import os
import json
import random
from utils import load_json

# pred
pred_tables = []
pred_columns = []

# files
querys = load_json("test_data/querys.json")
tables = load_json("test_data/tables.json")
label_tables = load_json("test_data/y_tables.json")
label_columns = load_json("test_data/y_columns.json")
path_to_save = "preds"

# random choice
for query in querys:
    db = [
        db
        for db in tables
        if (db["db_id"] == query["db_id"]) & (db["from"] == query["from"])
    ][0]
    # 从columns 里随机选
    table_names = db["table_names"]
    cols = db["column_names"][1:]
    cnt_cols = random.randint(2, len(cols) - 1)
    samples = random.sample(cols, cnt_cols)
    pred_table = []
    pred_column = []
    for sample in samples:
        table_name = table_names[sample[0]]
        column_name = "{}.{}".format(table_name, sample[1])
        pred_table.append(table_name)
        pred_column.append(column_name)
    pred_tables.append(sorted(set(pred_table)))
    pred_columns.append(sorted(set(pred_column)))
print(len(pred_tables))
print(len(pred_columns))

with open(os.path.join(path_to_save, "pred_tables.json"), "w", encoding="utf-8") as f:
    json.dump(pred_tables, f, ensure_ascii=False)

with open(os.path.join(path_to_save, "pred_columns.json"), "w", encoding="utf-8") as f:
    json.dump(pred_columns, f, ensure_ascii=False)
