from typing import Any, Optional

from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

from eval_retriever.chain_extract_sql.output_parser import OutputParser
from eval_retriever.chain_extract_sql.prompt import prompt_template


class ExtractSqlChain(LLMChain):
    """extract metadata from python code"""

    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = prompt_template
    input_key: str = "code"  #: :meta private:

    @property
    def _chain_type(self) -> str:
        return "extract"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = prompt_template,
        output_parser: BaseOutputParser = None,
        **kwargs: Any,
    ):
        if output_parser is None:
            output_parser = OutputParser()
        return cls(llm=llm, prompt=prompt, output_parser=output_parser, **kwargs)