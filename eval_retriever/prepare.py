import os
import json
from sql_metadata import Parser

# 待存数据集
my_dbs = []
my_querys = []
tables_true = []
columns_true = []


# 取两个 dev_gold.sql 提取 table 和 column
path_to_BIRD = "../datasets/BIRD_dev"
path_to_save = "test_data"

# 制作数据集、问题、正确答案（表格、结果）
with open(os.path.join(path_to_BIRD, "dev_tables.json")) as f:
    tables_bird = json.load(f)
with open(os.path.join(path_to_BIRD, "dev.json")) as f:
    querys_bird = json.load(f)

for query in querys_bird:
    parser = Parser(query["SQL"])
    try:
        db_id = query["db_id"]
        db = [x for x in tables_bird if x["db_id"] == db_id][0]
        table_true = parser.tables
        column_true = parser.columns  # columns 还不准,会漏
        if len(table_true) == 1:
            column_true = ["{}.{}".format(table_true[0], col) for col in column_true]
            db["from"] = "BIRD_dev"
            query["from"] = "BIRD_dev"
            my_dbs.append(db)
            my_querys.append(query)
            tables_true.append(table_true)
            columns_true.append(column_true)
        else:
            cnt = [col for col in column_true if "." in col]
            if len(cnt) == len(column_true):
                db["from"] = "BIRD_dev"
                query["from"] = "BIRD_dev"
                my_dbs.append(db)
                my_querys.append(query)
                tables_true.append(table_true)
                columns_true.append(column_true)
    except Exception as e:
        print(e)

path_to_SPIDER = "../datasets/spider"
with open(os.path.join(path_to_SPIDER, "tables.json")) as f:
    tables_spider = json.load(f)
with open(os.path.join(path_to_SPIDER, "dev.json")) as f:
    querys_spider = json.load(f)

for query in querys_spider:
    parser = Parser(query["query"])
    try:
        db_id = query["db_id"]
        db = [x for x in tables_spider if x["db_id"] == db_id][0]
        table_true = parser.tables
        column_true = parser.columns
        if len(table_true) == 1:
            column_true = ["{}.{}".format(table_true[0], col) for col in column_true]
            db["from"] = "SPIDER_dev"
            query["from"] = "SPIDER_dev"
            my_dbs.append(db)
            my_querys.append(query)
            tables_true.append(table_true)
            columns_true.append(column_true)
        else:
            cnt = [col for col in column_true if "." in col]
            if len(cnt) == len(column_true):
                db["from"] = "SPIDER_dev"
                query["from"] = "SPIDER_dev"
                my_dbs.append(db)
                my_querys.append(query)
                tables_true.append(table_true)
                columns_true.append(column_true)
    except Exception as e:
        print(e)

with open(os.path.join(path_to_save, "tables.json"), "w", encoding="utf-8") as f:
    json.dump(my_dbs, f, ensure_ascii=False)

with open(os.path.join(path_to_save, "querys.json"), "w", encoding="utf-8") as f:
    json.dump(my_querys, f, ensure_ascii=False)

with open(os.path.join(path_to_save, "y_tables.json"), "w", encoding="utf-8") as f:
    json.dump(tables_true, f, ensure_ascii=False)

with open(os.path.join(path_to_save, "y_columns.json"), "w", encoding="utf-8") as f:
    json.dump(columns_true, f, ensure_ascii=False)
