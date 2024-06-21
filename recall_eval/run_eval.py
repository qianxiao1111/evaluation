import os
import re
import ast
from recall_eval.prompt import (
    gen_sys_python,
    gen_sys_sql,
    gen_user,
    extract_sys_python,
    extract_sys_sql,
    extract_user,
)
from recall_eval.eval_metrics import Metric
from util import load_json, save_json


def format_inputs(samples, mode: str):
    # 按照 generate 的格式要求改造 inputs
    # 共有四种模式，分别为 sql代码生成、python 代码生成、sql结果抽取、python结果抽取
    assert mode in [
        "gen_sql",
        "gen_python",
        "extract_sql",
        "extract_python",
    ], "invalid format mode."
    # 把需要推理的数据拼成 message 形式
    msgs = []
    for sample in samples:
        if mode == "gen_sql":
            msg_sys = gen_sys_sql
            msg_user = gen_user.format(
                table_infos=sample["table_infos"], query=sample["query"]
            )
        elif mode == "gen_python":
            msg_sys = gen_sys_python
            msg_user = gen_user.format(
                table_infos=sample["table_infos"], query=sample["query"]
            )
        elif mode == "extract_sql":
            msg_sys = extract_sys_sql
            msg_user = extract_user.format(code=sample)
        elif mode == "extract_python":
            msg_sys = extract_sys_python
            msg_user = extract_user.format(code=sample)
        else:
            raise Exception("invalid format mode.")
        msg = [
            {"role": "system", "content": msg_sys},
            {"role": "user", "content": msg_user},
        ]
        msgs.append(msg)
    return msgs


def parser_text(text, mode: str):
    # 提取 llm_response['output_text'] 中的 生成代码 或 召回的表格和字段信息
    # 共有四种模式，分别为 sql代码生成、python 代码生成、sql结果抽取、python结果抽取
    assert mode in [
        "gen_sql",
        "gen_python",
        "extract_sql",
        "extract_python",
    ], "invalid format mode."
    if mode == "gen_sql":
        pattern = r"```sql\n(.*?)\n```"
        try:
            text = re.search(pattern, text, re.DOTALL).group(1)
            text = text.replace("\n", " ").strip()
        except Exception:
            text = ""
        return text
    elif mode == "gen_python":
        pattern = r"```python\n(.*?)\n```"
        try:
            text = re.search(pattern, text, re.DOTALL).group(1)
        except Exception:
            text = ""
        return text
    elif mode in [
        "extract_sql",
        "extract_python",
    ]:
        pattern_table = r"(?i)tables(?:\s+is)?\s*:\s*\[([^\]]+)\]"
        pattern_column = r"(?i)columns(?:\s+is)?\s*:\s*\[([^\]]+)\]"
        text = text.replace("【", "[").replace("】", "]").replace("`", "")
        match_tables = re.findall(pattern_table, text.strip())
        match_columns = re.findall(pattern_column, text.strip())
        tables = []
        columns = []
        if match_tables:
            try:
                tables = ast.literal_eval(f"[{match_tables[0]}]")
            except (SyntaxError, ValueError):
                pass
        if match_columns:
            try:
                columns = ast.literal_eval(f"[{match_columns[0]}]")
            except (SyntaxError, ValueError):
                pass
        return {"tables": tables, "columns": columns}
    else:
        raise Exception("invalid format mode.")


def parser_list(batch_resp, mode: str):
    # 批处理解析
    return [parser_text(resp["output_text"], mode) for resp in batch_resp]


def save_result(preds, report, test_file_path):
    # 保存 LLM 生成的内容 和 最终的 recall_eval 结果
    parent_path = os.path.dirname(test_file_path)
    save_path = os.path.join(parent_path, "recall_eval_llm_gen_data.json")
    save_json(save_path, preds)
    print(f"Recall Eval Saved:{save_path}")
    save_path = os.path.join(parent_path, "recall_eval_report.json")
    save_json(save_path, report)
    print(f"Recall Eval Saved:{save_path}")


def eval_outputs(preds, samples,lang=None):
    # 使用 Metric 中评估方法，评估表格、字段召回的相关指标
    def combine_metrics_under_key(pred_data, label_data, key):
        combined_metrics = {}
        for metric_name in ["averaged", "jaccard", "hamming"]:
            metric_results = getattr(Metric, metric_name)(pred_data, label_data)
            combined_metrics.update(metric_results)

        return {key: combined_metrics}

    pred_tables = [pred["tables"] for pred in preds]
    pred_columns = [pred["columns"] for pred in preds]
    label_tables = [sample["label_table"] for sample in samples]
    label_columns = [sample["label_col"] for sample in samples]
    if lang == "python":
        label_columns = [[col.split(".")[1] for col in cols] for cols in label_columns]
    table_metrics_combined = combine_metrics_under_key(
        pred_tables, label_tables, "table"
    )
    column_metrics_combined = combine_metrics_under_key(
        pred_columns, label_columns, "column"
    )
    merged_results = {**table_metrics_combined, **column_metrics_combined}
    return merged_results
