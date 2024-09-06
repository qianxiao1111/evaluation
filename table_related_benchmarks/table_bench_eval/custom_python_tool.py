import re 
import ast
from contextlib import redirect_stdout
from io import StringIO
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from typing import Optional
from langchain_core.callbacks.manager import CallbackManagerForToolRun


extra_functions = """
import numpy as np

def compare(list1, list2):
    # sort the list
    list1.sort()
    list2.sort()
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if np.isnan(list1[i]):
            if not np.isnan(list2[i]):
                return False
        elif list1[i] != list2[i]:
            return False
    return True


def std_digit(list_nums):
    new_list = []
    for i in range(len(list_nums)):
        new_list.append(round(list_nums[i], 2))
    return new_list


def compute_general_chart_metric(references, predictions):
    processed_references = []
    processed_predictions = []
    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)

    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    return compare(processed_references, processed_predictions)


def compute_pie_chart_metric(references, predictions):
    processed_references = []
    processed_predictions = []
    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)
    references = processed_references
    processed_references = []
    total = 0
    for reference in references:
        total += reference
    for reference in references:
        processed_references.append(round(reference / total, 2))

    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    return compare(processed_references, processed_predictions)


def get_line_y_predictions(plt):
    line_y_predctions = []
    lines = plt.gca().get_lines()
    line_y_predctions = [list(line.get_ydata()) for line in lines]
    return line_y_predctions


def get_bar_y_predictions(plt):
    bar_y_predctions = []
    patches = plt.gca().patches
    bar_y_predctions = [patch.get_height() for patch in patches]
    return bar_y_predctions


def get_hbar_y_predictions(plt):
    hbar_y_predctions = []
    patches = plt.gca().patches
    hbar_y_predctions = [patch.get_width() for patch in patches]
    return hbar_y_predctions


def get_pie_y_predictions(plt):
    pie_y_predctions = []
    patches = plt.gca().patches
    for patch in patches:
        theta1, theta2 = patch.theta1, patch.theta2
        value = round((theta2 - theta1) / 360.0, 2)
        pie_y_predctions.append(value)
    return pie_y_predctions


def get_area_y_predictions(plt):
    area_y_predctions = []
    area_collections = plt.gca().collections
    for area_collection in area_collections:
        area_items = []
        for item in area_collection.get_paths()[0].vertices[:, 1]:
            if item != 0:
                area_items.append(item)
        area_y_predctions.append(area_items)
    return list(area_y_predctions)


def get_radar_y_predictions(plt):
    radar_y_predctions = []
    radar_lines = plt.gca().get_lines()
    radar_y_predctions = [list(line.get_ydata()) for line in radar_lines]
    for i in range(len(radar_y_predctions)):
        radar_y_predctions[i] = radar_y_predctions[i][:-1]
    return radar_y_predctions


def get_scatter_y_predictions(plt):
    scatter_y_predctions = []
    scatter_collections = plt.gca().collections
    for scatter_collection in scatter_collections:
        scatter_items = []
        for item in scatter_collection.get_offsets():
            scatter_items.append(item[1])
        scatter_y_predctions.append(scatter_items)
    return scatter_y_predctions


def get_waterfall_y_predictions(plt):
    waterfall_y_predctions = []
    patches = plt.gca().patches
    waterfall_y_predctions = [patch.get_height() for patch in patches]
    return waterfall_y_predctions
"""

def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.

    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    query = "\n".join([extra_functions, query])
    return query

class CustomPythonTool(PythonAstREPLTool):

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            # 可选的输入处理
            if self.sanitize_input:
                query = sanitize_input(query)
                
            # 解析 AST 树
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            
            # 创建一个缓冲区来捕获print的输出
            io_buffer = StringIO()
            try:
                # 捕获执行期间的所有标准输出
                with redirect_stdout(io_buffer):
                    exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
                    module_end = ast.Module(tree.body[-1:], type_ignores=[])
                    module_end_str = ast.unparse(module_end)  # type: ignore
                    ret = eval(module_end_str, self.globals, self.locals)
                    
                    # 如果返回值是 None，返回捕获的输出；否则返回结果
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return io_buffer.getvalue() + str(ret)  # 同时返回输出和结果
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return io_buffer.getvalue()
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))

