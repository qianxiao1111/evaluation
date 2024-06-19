import re
from typing import Dict
from langchain_core.output_parsers.transform import BaseTransformOutputParser


def remove_pd_read_assignments(code):
    """
    使用正则表达式去掉形如 `variable = pd.read_xx('file.xxx')` 的赋值语句。

    参数:
    code (str): 需要处理的代码字符串。

    返回:
    str: 去掉指定赋值语句后的代码字符串。
    """
    # 正则表达式匹配形如 `任何变量名 = pd.read_任何函数名(任何字符串)` 的模式
    pattern = r"\b\w+\s*=\s*pd\.read_\w+\(.*?\)"
    result = re.sub(pattern, "", code)
    return result.strip()  # 去除首尾空白并返回结果


class SqlOutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    pattern_sql = r"```sql\n(.*?)\n```"

    def parse(self, text: str) -> Dict:
        try:
            sql = re.search(self.pattern_sql, text, re.DOTALL).group(1)
            sql = sql.replace("\n", " ").strip()
        except Exception:
            sql = ""
        return sql

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "gen"


class PyOutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    pattern_cot = r"Thought:(.*?)```"
    pattern_py = r"```python\n(.*?)\n```"

    def parse(self, text: str) -> Dict:
        try:
            cot = re.search(self.pattern_cot, text, re.DOTALL).group(1)
        except Exception:
            cot = ""
        try:
            py = re.search(self.pattern_py, text, re.DOTALL).group(1)
            py = remove_pd_read_assignments(py)  # 去除带 read 的语句
        except Exception:
            py = ""
        return cot, py, text

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "actor"


class InspectOutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    def parse(self, text: str) -> Dict:
        return text

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "gen"
