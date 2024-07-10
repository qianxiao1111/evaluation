

## 第一套
prompt_with_format_0 = """You have access to a number of pandas dataframes. Given a user question about the dataframes, write the Python code to answer it.

## Use the following format:
Question: The user question you need to answer.
Thought: Think about what you should do based on the provided DataFrames, the question, etc.
Python code: Generate python code, wrap it up with ```python ... ```.

## Here is the information about each dataframe

{df_info}

Question: {input}
"""

prompt_with_format_1 = """You have access to multiple pandas dataframes. To answer the user's question about the dataframes, write the necessary Python code.

## Use the following format:
Question: The user's query.
Thought: Analyze the provided dataframes and the question to determine the solution.
Python code: Generate the Python code, enclosed within ```python ... ```.

## Information on each dataframe is given below:

{df_info}

Question: {input}
"""

prompt_with_format_2 = """With access to several pandas dataframes, your task is to write Python code to address the user's question.

## Use the format below:
Question: The user's question.
Thought: Consider the question and the provided dataframes to determine the appropriate approach.
Python code: Provide the Python code, wrapped in ```python ... ```.

## Details about each dataframe:

{df_info}

Question: {input}
"""

prompt_with_format_3 = """You have several pandas dataframes at your disposal. Based on the user's question, write the Python code to answer it.

## Use this format:
Question: The user's question.
Thought: Analyze the question and the dataframes to decide on the best approach.
Python code: Write the Python code, enclosed in ```python ... ```.

## Information about each dataframe is provided below:

{df_info}

Question: {input}
"""

prompt_with_format_4 = """With several pandas dataframes available, your task is to write the Python code to address the user's question.

## Follow this format:
Question: The user's query.
Thought: Evaluate the dataframes and the question to determine the solution.
Python code: Generate the Python code, within ```python ... ```.

## Details about the dataframes:

{df_info}

Question: {input}
"""

prompt_with_format_list = [prompt_with_format_0, prompt_with_format_1, prompt_with_format_2, prompt_with_format_3, prompt_with_format_4]


## 第二套
prompt_with_instruction_0 = """
You have access to a number of pandas dataframes. Given a user question about the dataframes, write the Python code to answer it.

{df_info}

Question: {input}
"""

prompt_with_instruction_1 = """You have access to multiple pandas dataframes. To respond to the user's query about the dataframes, write the required Python code.

{df_info}

Question: {input}
"""

prompt_with_instruction_2 = """Given access to several pandas dataframes, write the Python code to answer the user's question.

{df_info}

Question: {input}
"""

prompt_with_instruction_3 = """You have access to several pandas dataframes. Using the provided information, write the Python code to respond to the user's question.

{df_info}

Question: {input}
"""

prompt_with_instruction_4 = """Given several pandas dataframes, write the Python code to answer the user's question.

{df_info}

Question: {input}
"""

prompt_with_instruction_list = [prompt_with_instruction_0, prompt_with_instruction_1, prompt_with_instruction_2, prompt_with_instruction_3, prompt_with_instruction_4]