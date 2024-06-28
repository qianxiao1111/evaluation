# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/25 15:50
@Auth ： zhaliangyu
@File ：utils.py
@IDE ：PyCharm
"""
import tokenize
from io import StringIO
import ast
import pandas as pd
import re
from typing import Any, Tuple
from langchain_experimental.tools.python.tool import PythonAstREPLTool

def extract_ori_observe(completion: str) -> str:
    # 正则表达式模式
    pattern = r"Observe：\n(.*?)\n\n"

    # 使用re.search进行匹配

    match = re.search(pattern, completion, re.DOTALL)
    return match.group(1)

def extract_code_without_comments(code):
    """
    从Python代码中提取除注释行以外的代码。

    :param code: str, 输入的Python代码
    :return: str, 提取后的代码
    """
    code_io = StringIO(code)
    result = []

    try:
        tokens = tokenize.generate_tokens(code_io.readline)
        for token_type, token_string, _, _, _ in tokens:
            # Skip comment tokens
            if token_type != tokenize.COMMENT:
                result.append(token_string)
    except tokenize.TokenError as e:
        print(f"Token error: {e}")

    return ''.join(result)


def is_python_code(line: str) -> bool:
    """Tool function for extract python code"""
    try:
        ast.parse(line)
        return True
    except:
        return False

def extract_text_before_code(text: str) -> str:
    """Tool function for extract text content"""
    lines = text.split('\n')
    text_before_code = []

    for line in lines:
        if is_python_code(line):
            break
        text_before_code.append(line)

    return '\n'.join(text_before_code)


def extract_python_code(text: str) -> str:
    """Tool function for extract python code"""
    lines = text.split('\n')
    python_code = []

    for line in lines:
        if is_python_code(line):
            python_code.append(line)

    return '\n'.join(python_code)

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def filter_cot(completion: str):
    """
    Filter the COT steps before python code
    :param completion: llm output contents
    :return filtered COT content
    """
    try:
        # 如果输出较为规范，可以使用这种方式提取cot部分的内容
        pattern = r"Thought:\s*(.*?)\s*(?=Python Code:)"
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            thought_content = match.group(1)
        else:
            # 如果输出内容相对杂乱
            thought_content = extract_text_before_code(completion)
        return thought_content
    except:
        return ""


def filter_code(completion: str) -> Tuple[str, str]:
    """
    Filter python code from the llm output completion
    :param completion: llm output contents
    :return filtered python code and execute code
    """

    try:
        # 输出形式符合prompt
        regex = r"```python\s(.*?)```"
        action_match = re.search(regex, completion, re.DOTALL)
        if action_match:
            action_input = action_match.group(1)
            action_input = action_input.strip(" ")
            action_input = action_input.strip('"')
            code = action_input.strip(" ")
        else:
            # 输出形式随意
            code = extract_python_code(completion)
            code = code.strip(" ")
        pure_code = extract_code_without_comments(code)
        return code, pure_code
    except:
        return "", ""


def get_tool(df: Any):
    """
    Define python code execute tool
    :param df: List[pd.DataFrame] or pd.DataFrame
    :return Runnable
    """
    tool = PythonAstREPLTool()
    if isinstance(df, pd.DataFrame):
        locals = {"df": df}
    else:
        locals = {}
        for i, dataframe in enumerate(df):
            locals[f"df{i + 1}"] = dataframe
    tool.locals = locals
    tool.globals = tool.locals
    return tool


def get_table_infos(
    df_paths: list[str],
) -> str:
    import pandas as pd
    from pathlib import Path

    table_infos = ""
    for i in range(len(df_paths)):
        path = df_paths[i]
        table_name = Path(path).stem
        df = pd.read_csv(path)
        df_head_markdown = df.head(3).to_markdown(index=False)
        table_infos += f"Table samples of {table_name}\n" + df_head_markdown + "\n"
    return table_infos