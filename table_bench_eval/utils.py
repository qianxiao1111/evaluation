import os 
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from table_bench_eval.custom_python_tool import CustomPythonTool

CODE_PREFIX = """import matplotlib.pyplot as plt
from mplfonts import use_font
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# Fixing Chinese font issues
use_font("Noto Serif CJK SC")\n"""

def valid_path(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def pre_save_table_to_csv(table):
    table_json = []
    for item in table['data']:
        row_data = {}
        for i in range(len(table['columns'])):
            row_data[table['columns'][i]] = item[i]
        table_json.append(row_data)
    df = pd.DataFrame(table_json)
    df.to_csv('table.csv', index=False)

def extract_final_answer(text):
    match = re.search(r'Final Answer:\s*(.*)', text)
    if match:
        return match.group(1).strip()
    return ""

def parse_final_answer_prediction(prediction):
    pattern = r"Final Answer: (.+)"
    try:
        match = re.search(pattern, prediction)
        if match:
            return match.group(1)
        else:
            return ''
    except Exception as e:
        return ''
        
def read_json_file(path, filter_func=None):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                if filter_func is not None:
                    json_data = list(filter(filter_func, json_data))
                return json_data
            except Exception as e:
                f.seek(0)
                lines = f.readlines()
                json_list = [json.loads(line.strip(
                )) for line in lines if filter_func is None or filter_func(json.loads(line.strip()))]
                return json_list
    else:
        return None


def write_json_to_file(path: str, data: dict, is_json_line: bool = False) -> None:
    valid_path(path)
    with open(path, 'w', encoding='utf-8') as f:
        if is_json_line:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))

def parse_python_code(prediction):
    pattern = r"```python\n(.*?)```"
    try:
        matches = re.findall(pattern, prediction, flags=re.S)
        if matches:
            return matches[-1]
        else:
            return ''
    except Exception as e:
        return ''

def get_tool(df: Any, df_names=None):
    """
    Define python code execute tool
    :param df: List[pd.DataFrame] or pd.DataFrame
    :return Runnable
    """
    tool = CustomPythonTool()
    if df_names == None:
        if isinstance(df, pd.DataFrame):
            locals = {"df": df}
        else:
            locals = {}
            for i, dataframe in enumerate(df):
                locals[f"df{i + 1}"] = dataframe
    else:
        locals = {}
        for i, dataframe in enumerate(df):
            locals[df_names[i]] = dataframe
    tool.locals = locals
    tool.globals = tool.locals
    return tool

def ensure_last_line_print(code):
    # 将代码按行分割
    lines = code.strip().split('\n')

    # 获取最后一行代码
    last_line = lines[-1].strip()

    # 检查最后一行是否已经包含 print 函数
    if not last_line.startswith('print'):

        # 尝试提取最后一行中的变量名或表达式
        # 这里假设最后一行是简单的变量赋值或表达式
        last_line_variable = last_line

        # 将变量包裹在print中
        lines[-1] = f'print({last_line_variable})'

    # 将所有行重新组合成代码字符串
    modified_code = '\n'.join(lines)
    return modified_code

def build_chart_eval_code(sample):
    answer = sample['answer']
    chart_type = sample['chart_type']
    prediction = sample['raw_generation'][0]

    python_code = parse_python_code(prediction)
    python_code = CODE_PREFIX + python_code

    # TestCase
    eval_code = '''
if chart_type == 'line':
    y_predictions = get_line_y_predictions(plt)
if chart_type == 'bar':
    y_predictions = get_bar_y_predictions(plt)
if chart_type == 'hbar':
    y_predictions = get_hbar_y_predictions(plt)
if chart_type == 'pie':
    y_predictions = get_pie_y_predictions(plt)
if chart_type == 'area':
    y_predictions = get_area_y_predictions(plt)
if chart_type == 'radar':
    y_predictions = get_radar_y_predictions(plt)
if chart_type == 'scatter':
    y_predictions = get_scatter_y_predictions(plt)
if chart_type == 'waterfall':
    y_predictions = get_waterfall_y_predictions(plt)

if chart_type == 'pie':
    print(compute_pie_chart_metric(y_references, y_predictions))
else:
    print(compute_general_chart_metric(y_references, y_predictions))
    '''
    # chart_eval_code = f'from chat_metric_utils import *\n{python_code}\n{answer}\nchart_type="{chart_type}"\n{eval_code}'
    chart_eval_code = f'{python_code}\n{answer}\nchart_type="{chart_type}"\n{eval_code}'

    if python_code == '':
        return '', ''
    return python_code, chart_eval_code

def parse_code_then_exec(prediction):
    ecr_1 = False
    python_code = parse_python_code(prediction)
    python_code = ensure_last_line_print(python_code)
    python_code = CODE_PREFIX + python_code
    df = pd.read_csv("table.csv")
    exec_tool = get_tool(df)
    try:
        observe = exec_tool.run(python_code)  # 需要监控超时的代码块
        if isinstance(observe, pd.DataFrame):
            observe = observe.head().to_markdown(index=False)
        else:
            observe = str(observe)
        ecr_1 = True
    except Exception as e:
        observe = e
    if observe != "":
        observe = observe.strip()
    return observe, ecr_1

def execution_eval(observe: str) -> bool:
    """
    Test whether the code generated by eval_llm can be executed.
    :param output: output code of llm generation
    :return: True or False
    """
    if observe == "": # 出来是空字符串， 返回False
        return False
    # 只要执行结果中不出现error 或者 exception， 就认为代码可执行
    pattern = re.compile(r"error|exception", re.IGNORECASE)
    try:
        res = not pattern.search(observe)
    except:
        res = True
    return res

def parse_chart_code_then_exec(sample):
    ecr_1 = False
    python_code, chart_eval_code = build_chart_eval_code(sample)
    try:
        df = pd.read_csv("table.csv")
        exec_tool = get_tool(df)
        _ = exec_tool.run(python_code)
        ecr_1 = True
    except Exception as e:
        pass
    try:
        df = pd.read_csv("table.csv")
        exec_tool = get_tool(df)
        observe = exec_tool.run(chart_eval_code)
        if isinstance(observe, pd.DataFrame):
            observe = observe.head().to_markdown(index=False)
        else:
            observe = str(observe)
    except Exception as e:
        observe = e
    observe = observe.strip()
    plt.close("all")
    return observe, ecr_1

def reformat_instruction(instruction: str) -> str:
    # Remove the phrase "in JSON format"
    modified_text = re.sub(r' in JSON format', '', instruction)
    # Replace the detailed JSON table content with a placeholder
    modified_text = re.sub(r'\[TABLE\]\s*\n\{.*\}', '{table_info}', modified_text)
    return modified_text

