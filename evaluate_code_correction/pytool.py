import ast


class DataFrameFlowVisitor(ast.NodeVisitor):
    """
    找到一段代码中最后一次使用到的 dataframe，包括赋值、计算、打印等方法。
    """

    def __init__(self):
        self.assignments = {}
        self.pandas_functions = {
            "DataFrame",
            "read_csv",
            "read_excel",
            "groupby",
            "sort_values",
            "merge",
            "pivot_table",
            "drop_duplicates",
            "fillna",
            "loc",
            "iloc",
        }
        self.last_df_variable = None
        self.pandas_attr_accesses = {
            "shape",
            "columns",
            "index",
            "values",
            "dtypes",
            "describe",
        }

    # 检测是否使用 函数，这是个辅助判断的节点
    def is_pandas_function_call(self, node):
        # Check if the call is a pandas function or method
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.pandas_functions:
                # Check if the function is called on a DataFrame variable
                return isinstance(node.func.value, ast.Name)
        return False

    # 检查是否有 Call 嵌套, 遇到 Subscript 就可以挺直了
    def is_nested_call(self, node):
        if isinstance(node, ast.Call):
            sub_node = node.func.value
            if isinstance(sub_node, ast.Call):
                return isinstance(sub_node.func, ast.Attribute)
            elif isinstance(sub_node, ast.Subscript):
                return True
            else:
                return False
        return False

    # 检查 Subscript 下是否有嵌套，遇到 Name 类型停止
    def is_nested_sub(self, node):
        if not isinstance(node, ast.Name):
            sub_node = node.func.value
            if isinstance(sub_node, ast.Name):
                return True
            else:
                return False
        else:
            return False

    def visit_Assign(self, node):
        # print(node.targets[0].id)

        # Direct DataFrame creation or modification
        sub_node = node.value
        # 对sub_node 做类型判断，并不断遍历到最末端的 node ,获得 id,判断 assignment
        while self.is_nested_call(sub_node):
            sub_node = sub_node.func.value

        # DataFrame function
        if (
            isinstance(sub_node, ast.Call)
            and isinstance(sub_node.func, ast.Attribute)
            and self.is_pandas_function_call(sub_node)
        ):
            self.do_update(node)

        # DataFrame subsetting
        elif isinstance(sub_node, ast.Subscript):
            sub_node = sub_node.value
            if isinstance(sub_node, ast.Call):
                while self.is_nested_sub(sub_node):
                    sub_node = sub_node.func.value
            if isinstance(sub_node, ast.Name):
                if sub_node.id in self.assignments:
                    self.do_update(node)

            # 针对 .loc .iloc
            elif (
                isinstance(sub_node, ast.Attribute)
                and sub_node.attr in self.pandas_functions
            ):
                self.do_update(node)
            else:
                pass

        # Assignment from another DataFrame variable
        elif isinstance(sub_node, ast.Name) and sub_node.id in self.assignments:
            self.do_update(node)

        else:
            # 还有一些没有考虑到的情况，无法枚举完
            pass

        self.generic_visit(node)

    def do_update(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id] = node
                self.update_last_df_variable(target.id)

    def update_last_df_variable(self, var_name):
        # 根据比较所在第几行更新
        if self.last_df_variable is None or self.get_lineno(
            self.assignments[var_name]
        ) >= self.get_lineno(self.assignments[self.last_df_variable]):
            self.last_df_variable = var_name

    def get_lineno(self, node):
        return node.lineno if hasattr(node, "lineno") else 0

    def find_last_relevant_dataframe(self):
        return self.last_df_variable


def extract_last_df(code, locals=None):
    if locals is not None:
        # 需要把 local  中的 df 引入
        # 加在最前面问题不大，因为不执行代码，只是解析语法
        code_example = ""
        for local in locals.keys():
            if "df" in local:
                code_example += "{} = pd.DataFrame()\n".format(local)
        code_example += code
    else:
        code_example = code
    # print("#-" * 50)
    # print(code_example)
    tree = ast.parse(code_example)
    visitor = DataFrameFlowVisitor()
    visitor.visit(tree)
    # 获取最后一个相关的DataFrame变量名
    last_df_variable = visitor.find_last_relevant_dataframe()

    return last_df_variable


format_result = """import pandas
def format_result(result):
    if type(result) is pandas.DataFrame:
        return list(result.itertuples(index=False, name=None))
    elif type(result) in [str, int, float]:
        resp = []
        temp = tuple()
        temp += (result,)
        resp.append(temp)
        return resp
    else:
        return []
"""
