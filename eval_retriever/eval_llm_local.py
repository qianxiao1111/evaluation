import os
import time
import json
import argparse
from langchain.schema.language_model import BaseLanguageModel
from langchain_openai import ChatOpenAI
from metrics import Metric
from gen import SqlGenChain, PyGenChain
from chain_extract_sql import ExtractSqlChain
from chain_extract_python import ExtractPythonChain
from utils import load_json, load_llm, extract_db_info, rename_columns
from util import start_service, is_service_up
import warnings

warnings.filterwarnings(action="ignore")


def gen_preds(
    llm_gen: BaseLanguageModel,
    llm_extract: BaseLanguageModel,
    num: int = None,
    path_to_dataset: str = None,
    path_to_save: str = ".",
):
    path_to_queries = os.path.join(path_to_dataset, "querys.json")
    path_to_tables = os.path.join(path_to_dataset, "tables.json")
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
    path_to_dataset: str = None,
):
    pred_tables = load_json(pred_tables_path)
    pred_columns = load_json(pred_columns_path)
    label_tables = load_json(os.path.join(path_to_dataset, "y_tables.json"))
    label_columns = load_json(os.path.join(path_to_dataset, "y_columns.json"))
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
    model_path = args.model_path
    max_len = args.max_len
    temperature = args.temperature if args.temperature else 0.01
    path_to_dataset = args.eval_dataset_path
    num = args.num
    path_to_save = args.eval_results_save_path
    model_kwargs = {}
    # 启动vllm 模型服务
    service_process, port, model_name = start_service(model_path, max_len)
    # 等待服务启动
    service_url = f"http://localhost:{port}"
    while not is_service_up(service_url):
        print("Waiting for the service to start...")
        time.sleep(3)
    time.sleep(2)
    print("服务已启动")
    # 业务代码
    service_openai_url = service_url + "/v1"
    llm = ChatOpenAI(
        temperature=temperature,
        openai_api_base=service_openai_url,
        openai_api_key="none",
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
    path_sql_tb, path_sql_col, path_py_tb, path_py_col = gen_preds(
        llm_gen=llm,
        llm_extract=llm,
        num=num,
        path_to_dataset=path_to_dataset,
        path_to_save=path_to_save,
    )
    result = {}
    result["sql"] = evaluate_metrics(path_sql_tb, path_sql_col, num, path_to_dataset)
    result["python"] = evaluate_metrics(path_py_tb, path_py_col, num, path_to_dataset)
    print(json.dumps(result, indent=4))
    service_process.terminate()
    with open(os.path.join(path_to_save, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics.")
    parser.add_argument(
        "--model_path",
        required=False,
        default="/home/dev/weights/CodeQwen1.5-7B-Chat",
        help="llm model path",
    )
    parser.add_argument(
        "--max_len", type=int, required=False, default=8192, help="max seq length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.01,
        help="temperature of llm",
    )
    parser.add_argument(
        "--eval_dataset_path",
        required=False,
        default="../evalset",
        help="folder to dataset",
    )
    parser.add_argument(
        "--eval_results_save_path",
        required=False,
        default="../evalset",
        help="folder to save",
    )
    parser.add_argument(
        "--num", type=int, required=False, default=None, help="number of lines to eval"
    )
    args = parser.parse_args()
    main(args)
