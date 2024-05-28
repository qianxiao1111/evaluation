import json
from langchain.llms import HuggingFaceTextGenInference
from langchain_core.language_models.llms import LLM


key_query = {
    "SPIDER_dev": "query",
    "BIRD_dev": "SQL",
}


def rename_columns(columns, db_info):
    resp = []
    for column in columns:
        tmp = [
            "{}.{}".format(db_info["table_names"][info[0]], column)
            for info in db_info["column_names"]
            if info[1] == column
        ]
        resp.extend(tmp)
    return resp


def extract_db_info(db_info):
    # 提取并格式化表信息
    tableinfo = {}
    for i, table_name in enumerate(db_info["table_names"]):
        column_list = [col[1] for col in db_info["column_names"] if col[0] == i]
        tableinfo[table_name] = column_list

    # 外键信息处理
    foreign_keys = db_info["foreign_keys"]
    foreign_keys_info = []
    for fk in foreign_keys:
        idx0, idx1 = fk
        fk0 = db_info["column_names"][idx0]
        fk1 = db_info["column_names"][idx1]
        foreign_keys_info.append(
            "{}.{} = {}.{}".format(
                db_info["table_names"][fk0[0]],
                fk0[1],
                db_info["table_names"][fk1[0]],
                fk1[1],
            )
        )

    return "table:\n{}\nforeign_keys:{}\n".format(tableinfo, foreign_keys_info)


def load_json(path_to_json: str):
    with open(path_to_json, "r", encoding="utf-8") as f:
        resp = json.load(f)
    return resp


def load_llm(infer_url: str, temperature: float = 0.01, model: str = None) -> LLM:
    assert model in ["hf", "deepseek", "qwen"], "Invalid model."
    if model == "hf":
        return HuggingFaceTextGenInference(
            inference_server_url=infer_url,
            max_new_tokens=1024,
            temperature=temperature,
            repetition_penalty=1.03,
            seed=42,
            stop_sequences=[
                "\nNew_Question:",
                "\nQuestion:",
                "\nTable",
            ],
        )
    elif model == "deepseek":
        return ChatOpenAI(
            temperature=temperature,
            max_tokens=1024,
            verbose=True,
            openai_api_base="{}/v1".format(infer_url),
            openai_api_key="none",
            model_name="deepseek-coder-6.7b-instruct",
        )
    elif model == "qwen":
        return ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="{}/v1".format(infer_url),
            model_name="qwen1.5-14b-awq",
            max_tokens=1024,
            temperature=temperature,
        )
    else:
        return None
