import re

from langchain_core.output_parsers.transform import BaseTransformOutputParser


class OutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    pattern_table = r"the tables is \[([^\]]+)\]"
    pattern_column = r"the columns is \[([^\]]+)\]"

    def parse(self, text: str) -> str:
        tables = []
        columns = []
        text = text.replace("【", "[").replace("】", "]").replace("`", "")
        match_tables = re.findall(self.pattern_table, text.strip())
        match_columns = re.findall(self.pattern_column, text.strip())
        if match_tables:
            tables = [t.strip() for t in match_tables[0].split(",")]
        if match_columns:
            columns = [c.strip() for c in match_columns[0].split(",")]
        return {"tables": tables, "columns": columns}

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "extract"
