messages_format = [
    {
        "role": "system",
        "content": "You are 数海闻涛, an expert Python data analyst developed by 浙江大学计算机创新技术研究院 (Institute of Computer Innovation of Zhejiang University, or ZJUICI). Your job is to help user analyze datasets by writing Python code. Each markdown codeblock you write will be executed in an IPython environment, and you will receive the execution output. You should provide results analysis based on the execution output.\nFor politically sensitive questions, security and privacy issues, or other non-data analyze questions, you will refuse to answer.\n\nRemember:\n- Comprehend the user's requirements carefully & to the letter.\n- If additional information is needed, feel free to ask the user.\n- Give a brief description for what you plan to do & write Python code.\n- You can use `read_df(uri: str) -> pd.DataFrame` function to read different file formats into DataFrame.\n- When creating charts, prefer using `seaborn`.\n- If error occurred, try to fix it.\n- Response in the same language as the user.\n- Today is 2024-08-26",
    },
    {
        "role": "user",
        "content": "文件名称: '{table_name}'",
    },
    {
        "role": "assistant",
        "content": "我已经收到您的数据文件，我需要查看文件内容以对数据集有一个初步的了解。首先我会读取数据到 `{df_name}` 变量中，并通过 `{df_name}.info` 查看 NaN 情况和数据类型。\n\n```python\n# Load the data into a DataFrame\n{df_name} = read_df('{table_name}')\n\n# Remove leading and trailing whitespaces in column names\n{df_name}.columns = {df_name}.columns.str.strip()\n\n# Remove rows and columns that contain only empty values\n{df_name} = {df_name}.dropna(how='all').dropna(axis=1, how='all')\n\n# Get the basic information of the dataset\n{df_name}.info(memory_usage=False)\n```",
    },
    {
        "role": "system",
        "content": "```pycon\n{df_info}\n```",
    },
    {
        "role": "assistant",
        "content": "接下来我将用 `{df_name}.head(5)` 来查看数据集的前 5 行。\n\n```python\n# Show the first 5 rows to understand the structure\n{df_name}.head(5)\n```",
    },
    {
        "role": "system",
        "content": "```pycon\n{df_head}\n```",
    },
    {
        "role": "assistant",
        "content": "我已经了解了数据集 {table_name} 的基本信息。请问我可以帮您做些什么？",
    },
    # {
    #     "role": "user",
    #     "content": "分析不同学历的投保人的理赔失败总数量，并绘制柱状图展示。",
    # },
]

import pandas as pd
from io import StringIO
import json
from openai import OpenAI
import re
from langchain_experimental.tools import PythonAstREPLTool

pd.set_option("display.width", 2048)
# 8 is the minimum value to display `df.describe()`. We have other truncation mechanisms so it's OK to flex this a bit.
pd.set_option("display.max_rows", 8)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.precision", 3)

CODE_PREFIX = """
import pandas as pd
import numpy as np
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
"""


def rebuild_messages(messages_format, table_paths, query):
    messages = []
    messages.append(messages_format[0])
    for i, table_path in enumerate(table_paths):
        table_name = table_path.split("/")[-1]
        df_name = f"df{i+1}"

        # read df
        if table_path.endswith(".xlsx"):
            df = pd.read_excel(table_path)
        elif table_path.endswith(".csv"):
            df = pd.read_csv(table_path)
        else:
            raise ValueError("Only supports data in CSV and XLSX formats.")

        # data preprocess
        df.columns = df.columns.str.strip()
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # capture df info
        buffer = StringIO()
        df.info(memory_usage=False, buf=buffer)
        df_info = buffer.getvalue()
        df_head = df.head(5)
        for mes in messages_format[1:]:
            mes["content"] = mes["content"].format(
                table_name=table_name, df_info=df_info, df_head=df_head, df_name=df_name
            )
            messages.append({"role": mes["role"], "content": mes["content"]})
    messages.append({"role": "user", "content": query})
    # print(messages)
    # exit()
    return messages


def get_client():
    # client = OpenAI(api_key="****")
    # completion = client.chat.completions.create(
    #     # model="gpt-4o",
    #     model="gpt-4o-mini",
    #     messages=messages,
    #     # stop=["\nObservation:"],
    #     temperature=0.01,
    #     timeout=40,
    # )

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8080/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    return client


def excute_code(text, tabel_paths):
    # extract python code
    regex = r"```python\s(.*?)```"
    action_match = re.search(regex, text, re.DOTALL)
    if action_match:
        action_input = action_match.group(1)
        action_input = action_input.strip(" ")
        action_input = action_input.strip('"')
        code = action_input.strip(" ")
    else:
        # 输出形式随意
        code = text.strip(" ")

    # define local var
    locals_var = {}
    for i, table_path in enumerate(tabel_paths):
        df_name = f"df{i+1}"
        # read df
        if table_path.endswith(".xlsx"):
            df = pd.read_excel(table_path)
        elif table_path.endswith(".csv"):
            df = pd.read_csv(table_path)
        else:
            raise ValueError("Only supports data in CSV and XLSX formats.")
        locals_var[df_name] = df
    tool = PythonAstREPLTool()
    tool.locals = locals_var
    tool.globals = tool.locals

    if "plt" in code:
        code = code + "\nplt.savefig('img.png',bbox_inches='tight')"
    obs = tool.run(CODE_PREFIX + code)

    print("code excute result:")
    print(obs)


if __name__ == "__main__":
    # vllm api server，线上显卡v100dtype是half，跟bf16结果不同。
    #CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model /data4/sft_output/qwen2-base-0817/final/ --served-model-name qwen2-7b-sft --port 8080 --max-model-len 32768 --dtype half --gpu-memory-utilization 0.95
    table_paths = ["/data4/evaluation/output/测试数据/small_fund_table.csv"]
    query = ".近一周涨跌幅超过10%的基金中，有多少个基金的成立时间在2018年及之前？"
    messages = rebuild_messages(messages_format, table_paths, query)
    client = get_client()

    chat_response = client.chat.completions.create(
        model="qwen2-7b-sft",
        messages=messages,
        top_p=0.3,
        temperature=0.1,
        max_tokens=1024,
    )
    res = chat_response.choices[0].message.content
    print("Chat response:")
    print(res)

    excute_code(res, table_paths)
