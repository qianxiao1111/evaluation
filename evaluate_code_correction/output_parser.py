# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/27 13:10
@Auth ： zhaliangyu
@File ：output_parser.py
@IDE ：PyCharm
"""
# -*- coding: utf-8 -*-

import re
from typing import Union

from langchain.agents.mrkl.output_parser import (
    FINAL_ANSWER_ACTION,
    MRKLOutputParser,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

TOOL_PREFIX = """import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.max_rows', 6)
"""


class CustomOutputParser(MRKLOutputParser):
    includes_answer: bool = False
    tool_name: str = "python_repl_tool"

    def __init__(self, tool_name: str):
        super().__init__(tool_name=tool_name)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        self.includes_answer = FINAL_ANSWER_ACTION in text

        FINAL_ANSWER_ACTION_CHINESE = [
            "最终答案",
            "最终答案是",
            "最终回答是",
            "最终结果是",
            "最后结果",
            "final answer",
            "Final answer"
        ]
        # replace possible chinese action expressions to FINAL_ANSWER_ACTION
        for answer_action in FINAL_ANSWER_ACTION_CHINESE:
            if answer_action in text:
                text = text.replace(answer_action, FINAL_ANSWER_ACTION)
                self.includes_answer = True
                break

        regex = r"```python\s(.*?)```"
        if "pd.datetime" in text:
            text = text.replace("pd.datetime", "datetime")
        action_match = re.search(regex, text, re.DOTALL)
        if action_match and self.includes_answer:
            if text.find(FINAL_ANSWER_ACTION) < text.find(action_match.group(0)):
                # if final answer is before the hallucination, return final answer
                start_index = text.find(FINAL_ANSWER_ACTION) + len(FINAL_ANSWER_ACTION)
                end_index = text.find("\n\n", start_index)
                return AgentFinish(
                    {"output": text[start_index:end_index].strip()}, text[:end_index]
                )
            else:
                start_index = text.find(FINAL_ANSWER_ACTION)
                return AgentFinish(
                    {"output": text[start_index:].strip()}, text[start_index:].strip()
                )

        if action_match:
            action = self.tool_name
            # action_input = action_match.group(2)
            action_input = action_match.group(1)
            action_input = action_input.strip(" ")
            action_input = action_input.strip('"')
            tool_input = action_input.strip(" ")

            tool_prefix = TOOL_PREFIX
            tool_input = tool_prefix + tool_input
            # if there is no print func in the last line of code, add to it
            last_line = tool_input.rstrip("\n").split("\n")[-1]
            if "print" not in last_line:
                try:
                    if "=" in last_line:
                        var = last_line.split("=")[0].strip()
                        code_suffix = f"print({var})"
                        tool_input += code_suffix
                    elif "#" not in last_line:
                        var = last_line.strip()
                        code_suffix = f"print({var})"
                        tool_input += code_suffix
                    else:
                        if "final_df" in tool_input:
                            code_suffix = "print(final_df)"
                            tool_input += code_suffix
                except:
                    pass

            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif self.includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(
            r"```python(.*)```",
            text,
            re.DOTALL,
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Invalid Format: Missing 'python code input' after 'Thought:'",
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
