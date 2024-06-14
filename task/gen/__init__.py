from typing import Any, Optional, List
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain_core.prompts import BasePromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
from task.gen.prompt import (
    prompt_inspect,
    prompt_template_sql,
    prompt_template_py,
    prompt_template_regen,
)
from task.gen.output_parser import InspectOutputParser, SqlOutputParser, PyOutputParser


class SqlGenChain(LLMChain):
    """generate table column comment"""

    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = prompt_template_sql
    input_key: List = ["table_infos", "query"]

    @property
    def _chain_type(self) -> str:
        return "gen"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = prompt_template_sql,
        output_parser: BaseOutputParser = None,
        **kwargs: Any,
    ):
        if output_parser is None:
            output_parser = SqlOutputParser()
        return cls(llm=llm, prompt=prompt, output_parser=output_parser, **kwargs)


class PyGenChain(LLMChain):
    """generate table column comment"""

    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = prompt_template_py
    input_key: List = ["table_infos", "query"]

    @property
    def _chain_type(self) -> str:
        return "gen"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = prompt_template_py,
        output_parser: BaseOutputParser = None,
        **kwargs: Any,
    ):
        if output_parser is None:
            output_parser = PyOutputParser()
        return cls(llm=llm, prompt=prompt, output_parser=output_parser, **kwargs)


class InspectChain(LLMChain):
    """generate table column comment"""

    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = prompt_inspect
    input_key: List = ["table_infos", "query"]

    @property
    def _chain_type(self) -> str:
        return "gen"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = prompt_inspect,
        output_parser: BaseOutputParser = None,
        **kwargs: Any,
    ):
        if output_parser is None:
            output_parser = InspectOutputParser()
        return cls(llm=llm, prompt=prompt, output_parser=output_parser, **kwargs)


class RegenChain(LLMChain):
    """generate table column comment"""

    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = prompt_template_py
    input_key: List = ["table_infos", "query"]

    @property
    def _chain_type(self) -> str:
        return "gen"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = prompt_template_regen,
        output_parser: BaseOutputParser = None,
        **kwargs: Any,
    ):
        if output_parser is None:
            output_parser = PyOutputParser()
        return cls(llm=llm, prompt=prompt, output_parser=output_parser, **kwargs)
