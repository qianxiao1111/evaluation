import re

from langchain_core.output_parsers.transform import BaseTransformOutputParser


class OutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    pattern_table = r"tables is: \[([^\]]+)\]"
    pattern_column = r"columns is: \[([^\]]+)\]"

    def parse(self, text: str) -> str:
        # print(text)
        tables = []
        columns = []
        text = text.replace("【", "[").replace("】", "]").replace("`", "")
        match_tables = re.findall(self.pattern_table, text.strip())
        match_columns = re.findall(self.pattern_column, text.strip())
        if match_tables:
            tables = eval("[{}]".format(match_tables[0]))
        if match_columns:
            columns = eval("[{}]".format(match_columns[0]))
        return {"tables": tables, "columns": columns}

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "extract"
