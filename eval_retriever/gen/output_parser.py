import re
from typing import Dict
from langchain_core.output_parsers.transform import BaseTransformOutputParser


class OutputParser(BaseTransformOutputParser[str]):
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

    pattern_py = r"```python\n(.*?)\n```"

    def parse(self, text: str) -> Dict:
        try:
            py = re.search(self.pattern_py, text, re.DOTALL).group(1)
        except Exception:
            py = ""
        return py

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "gen"
