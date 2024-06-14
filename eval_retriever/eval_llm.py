import os
import json
import argparse
from langchain.schema.language_model import BaseLanguageModel
from metrics import Metric
from gen import SqlGenChain, PyGenChain
from chain_extract_sql import ExtractSqlChain
from chain_extract_python import ExtractPythonChain
from utils import load_json, load_llm, extract_db_info, rename_columns
import warnings

warnings.filterwarnings(action="ignore")


def gen_preds(
    llm_gen: BaseLanguageModel,
    llm_extract: BaseLanguageModel,
    num: int = None,
    path_to_queries: str = "datasets/evalset/querys.json",
    path_to_tables: str = "datasets/evalset/tables.json",
    path_to_save: str = "report",
):
    samples = load_json(path_to_queries)
    tables = load_json(path_to_tables)
    sql_pred_tables = []
    py_pred_tables = []
    sql_pred_columns = []
    py_pred_columns = []
    if num is not None:
        samples = samples[:num]
    sql_gen_chain = SqlGenChain.from_llm(llm=llm_gen)
    py_gen_chain = PyGenChain.from_llm(llm=llm_gen)
    sql_extract_chain = ExtractSqlChain.from_llm(llm=llm_extract)
    py_extract_chain = ExtractPythonChain.from_llm(llm=llm_extract)
    for sample in samples:
        question = sample["question"]
        print("=" * 200)
        print(question)
        db_info = [
            db
            for db in tables
            if (db["db_id"] == sample["db_id"]) & (db["from"] == sample["from"])
        ][0]
        # sql
        table_infos = extract_db_info(db_info)
        sql_code = sql_gen_chain.predict(table_infos=table_infos, query=question)
        print("\n")
        print("SQL:\n```\n{}\n```".format(sql_code))
        retract_sql = sql_extract_chain.predict(code=sql_code)
        sql_pred_tables.append(retract_sql["tables"])
        sql_pred_columns.append(retract_sql["columns"])
        # python
        py_code = py_gen_chain.predict(table_infos=table_infos, query=question)
        print("\n")
        print("Python:\n```\n{}\n```".format(py_code))
        retract_py = py_extract_chain.predict(code=py_code)
        # columns 需要加上 tablename
        retract_py["columns"] = rename_columns(retract_py["columns"], db_info)
        py_pred_tables.append(retract_py["tables"])
        py_pred_columns.append(retract_py["columns"])

    with open(
        os.path.join(path_to_save, "sql_pred_tables.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(sql_pred_tables, f, ensure_ascii=False)

    with open(
        os.path.join(path_to_save, "sql_pred_columns.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(sql_pred_columns, f, ensure_ascii=False)

    with open(
        os.path.join(path_to_save, "py_pred_tables.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(py_pred_tables, f, ensure_ascii=False)

    with open(
        os.path.join(path_to_save, "py_pred_columns.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(py_pred_columns, f, ensure_ascii=False)

    return (
        os.path.join(path_to_save, "sql_pred_tables.json"),
        os.path.join(path_to_save, "sql_pred_columns.json"),
        os.path.join(path_to_save, "py_pred_tables.json"),
        os.path.join(path_to_save, "py_pred_columns.json"),
    )


def evaluate_metrics(
    pred_tables_path: str,
    pred_columns_path: str,
    num: int = None,
    label_tables_path: str = "datasets/evalset/y_tables.json",
    label_columns_path: str = "datasets/evalset/y_columns.json",
):
    pred_tables = load_json(pred_tables_path)
    pred_columns = load_json(pred_columns_path)
    label_tables = load_json(label_tables_path)
    label_columns = load_json(label_columns_path)
    if num is not None:
        pred_tables = pred_tables[:num]
        pred_columns = pred_columns[:num]
        label_tables = label_tables[:num]
        label_columns = label_columns[:num]

    def combine_metrics_under_key(pred_data, label_data, key):
        combined_metrics = {}
        for metric_name in ["averaged", "jaccard", "hamming"]:
            metric_results = getattr(Metric, metric_name)(pred_data, label_data)
            combined_metrics.update(metric_results)

        return {key: combined_metrics}

    table_metrics_combined = combine_metrics_under_key(
        pred_tables, label_tables, "table"
    )
    column_metrics_combined = combine_metrics_under_key(
        pred_columns, label_columns, "column"
    )
    merged_results = {**table_metrics_combined, **column_metrics_combined}
    return merged_results


def main(args):
    llm_gen = load_llm(infer_url=args.gen_model_url, model="hf")
    llm_extract = load_llm(infer_url=args.gen_model_url, model="hf")
    path_sql_tb, path_sql_col, path_py_tb, path_py_col = gen_preds(
        llm_gen=llm_gen,
        llm_extract=llm_extract,
        num=int(args.num),
        path_to_save=args.path_to_save,
    )
    result = {}
    result["sql"] = evaluate_metrics(
        path_sql_tb,
        path_sql_col,
        args.num,
    )
    result["python"] = evaluate_metrics(
        path_py_tb,
        path_py_col,
        args.num,
    )
    print(json.dumps(result, indent=4))
    with open(
        os.path.join(args.path_to_save, "result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics.")
    parser.add_argument(
        "--gen_model_url",
        required=True,
        help="http url for llm inference, this is the eval llm server url link",
    )
    parser.add_argument(
        "--extract_model_url",
        required=True,
        help="http url for llm inference, this is the anwser extract llm server url link",
    )
    parser.add_argument(
        "--path_to_save",
        required=False,
        default="datasets/report",
        help="folder to save",
    )
    parser.add_argument(
        "--num", type=int, required=False, help="number of lines to eval"
    )
    args = parser.parse_args()
    main(args)
