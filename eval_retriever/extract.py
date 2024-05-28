from typing import Dict
from sql_metadata import Parser
from chain_extract_sql import ExtractSqlChain
from chain_extract_python import ExtractPythonChain


def parser_from_sql(sql: str) -> Dict:
    parser = Parser(sql)
    return {"table": parser.tables, "column": parser.columns}


def extract_from_sql(code: str, chain: ExtractSqlChain) -> Dict:
    return chain.predict(code=code)


def extract_from_python(code: str, chain: ExtractPythonChain) -> Dict:
    return chain.predict(code=code)
