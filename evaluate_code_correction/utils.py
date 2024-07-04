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
from evaluate_code_correction.pytool import format_result, extract_last_df


def recraft_query(query, locals):
    last_df = extract_last_df(query, locals)
    end_str = "\n" + format_result + "print(format_result({}))".format(last_df)
    recraft_query = query + end_str
    return recraft_query


def extract_ori_observe(completion: str) -> str:
    # 正则表达式模式
    pattern = r"(Observe):\n(.*?)\n\n"

    # 使用re.search进行匹配

    match = re.search(pattern, completion, re.DOTALL)
    return match.group(1)


def extract_code_without_comments(code):
    """
    从Python代码中提取除注释行以外的代码。

    :param code: str, 输入的Python代码
    :return: str, 提取后的代码
    """
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"'''[\s\S]*?'''", "", code)

    # 移除单行注释
    lines = code.split("\n")
    cleaned_lines = []
    for line in lines:
        # 移除以 # 开始的注释，但保留字符串中的 #
        cleaned_line = re.sub(r'(?<!["\'"])#.*$', "", line)
        cleaned_lines.append(cleaned_line.rstrip())  # rstrip() 移除行尾空白
    # 重新组合代码，保留空行以维持原始结构
    return "\n".join(cleaned_lines)


def is_python_code(line: str) -> bool:
    """Tool function to check if a line of text is Python code"""
    try:
        tree = ast.parse(line)
        # Check if the parsed tree has at least one node that represents executable code
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expr, ast.Assign, ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.For, ast.While, ast.If, ast.With, ast.Raise, ast.Try)):
                return True
        return False
    except SyntaxError:
        return False


def extract_text_before_code(text: str) -> str:
    """Tool function for extract text content"""
    lines = text.split("\n")
    text_before_code = []

    for line in lines:
        if is_python_code(line):
            break
        text_before_code.append(line)

    return "\n".join(text_before_code)


def extract_python_code(text: str) -> str:
    """Tool function for extract python code"""
    lines = text.split("\n")
    python_code = []

    for line in lines:
        if is_python_code(line):
            python_code.append(line)

    return "\n".join(python_code)


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


# def get_table_infos(
#     df_paths: list[str],
# ) -> str:
#     import pandas as pd
#     from pathlib import Path


#     if len(df_paths) == 1:
#         df = pd.read_csv(df_paths[0])
#         df_head_markdown = df.head(3).to_markdown(index=False)
#         table_infos = f"Table samples of df\n" + df_head_markdown + "\n"
#     else:
#         table_infos = ""
#         for i in range(len(df_paths)):
#             path = df_paths[i]
#             table_name = Path(path).stem
#             df = pd.read_csv(path)
#             df_head_markdown = df.head(3).to_markdown(index=False)
#             table_infos += f"Table samples of df{i+1}\n" + df_head_markdown + "\n"
#     return table_infos


def get_table_infos(table_paths):
    """将所有csv文件对应的df-info拼装到一起"""
    infos_list = []
    if len(table_paths) == 1:
        df_markdown_info = str(
            pd.read_csv(table_paths[0], encoding="utf-8").head(3).to_string(index=False)
        )
        normalized_head = f"""/*\n"df.head()" as follows:\n{df_markdown_info}\n*/"""
        infos_list.append(normalized_head)
    else:
        for i, path in enumerate(table_paths):
            # normalized_name = normalize_table_name(path)
            df_markdown_info = str(
                pd.read_csv(path, encoding="utf-8").head(3).to_markdown(index=False)
            )
            normalized_head = (
                f"""/*\n"df{i+1}.head()" as follows:\n{df_markdown_info}\n*/"""
            )
            # single_table_name = "\n".join([normalized_head, df_markdown_info])
            infos_list.append(normalized_head)
    return "\n".join(infos_list)
