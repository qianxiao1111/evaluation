import os
import glob
import json
import sqlite3
import pandas as pd
from tqdm import tqdm

benches = {
    "BIRD_dev": "datasets/BIRD_dev/dev_databases",
    "SPIDER_dev": "datasets/spider/database",
}
path_to_save = "datasets/csv_lower/{}/{}.csv"


def get_names(
    conn,
):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    cursor.close()
    return [tb[0] for tb in table_names]


exist = []
for bench, bench_folder in benches.items():
    db_files = glob.glob(os.path.join(bench_folder, "**", "*.sqlite"), recursive=True)
    for db_file in tqdm(db_files):
        db_name = "{}-{}".format(bench, db_file.split("/")[-1].split(".sqlite")[0])
        print(db_name)
        if not os.path.exists("datasets/csv_lower/{}".format(db_name)):
            os.mkdir("datasets/csv_lower/{}".format(db_name))
        if db_name in exist:
            raise Exception("duplicat db name")
        exist.append(db_name)
        # start
        conn = sqlite3.connect(db_file)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        table_names = get_names(conn)
        for table_name in table_names:
            df = pd.read_sql_query("SELECT * FROM `{}`;".format(table_name), conn)
            df.to_csv(path_to_save.format(db_name, table_name.lower()), index=False)
