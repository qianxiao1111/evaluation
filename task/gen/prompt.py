from langchain.prompts import PromptTemplate

TEMPLATE_SQL = """You are an sql-code writer, write sql code based on below information:
{table_infos}

Question:
{query}


follow the plan write the sql query 

sql:
```sql
select related table and columns, and use correct sql query to solve problem
```

Answer in format:"""
prompt_template_sql = PromptTemplate(
    input_variables=["table_infos", "query"],
    template=TEMPLATE_SQL,
)


TEMPLATE_PY = """You are a python code writer, write python code based on below information:
{table_infos}

<format>
Question: question based on tables
Thought: plans to solve the problem

```python
import numpy as np
import pandas as pd

# Data Preparation: This may include creating new columns, converting test_data types, etc.

# Data Processing: This may include grouping, filtering, etc.

# Declare `final_df` Variable: Assign the prepared and processed test_data to `final_df`.

# Print the final result based on the question

```
</format>

Begin

<format>
Question:{query}
"""
prompt_template_py = PromptTemplate(
    input_variables=["table_infos", "query"],
    template=TEMPLATE_PY,
)

PROMPT_INSPECT = """
You are an inspector for code execution results. A question based on a table has been presented:

Question:
{question}

Table Head:
{table_infos}

Execution Output:
{exec}

Correct Answer:
{answer}

Please assess whether the execution outcome aligns with the provided correct answer. 
Respond with `True` if it matches, and `False` otherwise. 
Do no generate additional information.

begin
Answer:"""

prompt_inspect = PromptTemplate(
    input_variables=["question", "table_infos", "exec", "answer"],
    template=PROMPT_INSPECT,
)


RECTIFY_PROMPT_PYTHON = """
你现在正在充当一名Python代码reviewer，你需要根据输入的query、表格信息、运行的错误信息来对代码进行校正。

输出的内容需要保持以下格式：
<format>
Thought: 思考解决问题的步骤
Python Code:
```python
# Data preparation: 这一步可能包括创建新列、转换数据类型等。

# Data processing: 这一步可能包括分组、过滤等。

# Declare `final_df` var: 将经过准备和处理后的数据分配给`final_df`。

# Print the final result based on the question: 打印符合题意的最终结果
```
</format>

以下是输入的表格信息：
{table_infos}

以下是输入的query：
{query}

以下是原始的Thought以及Python代码：
{output}

以下是原始Python代码运行后的结果Observe：
{observe}

开始！
请根据以上约定的格式, 对输入的代码进行校正:\n

"""

CLASSIFY_PROMPT_PYTHON = """
你现在正在充当一名Python代码reviewer，输入的代码以及执行结果是根据用户的query和表格信息生成的。
但由于程序员水平有限，这个输入的代码可能是错的。 你需要根据用户的query/表格信息以及真实结果，来对代码以及代码执行结果的正确性进行判断。

以下是输入的query
{query}

以下是表格信息
{table_infos}

以下是程序员根据query和表格信息所生成的代码
{code}

以下是代码的执行观测结果：
{observation}

以下是用户问题对应的真实结果：
{true_result}

请根据以上内容，判定代码的正确性。
如果认为代码是正确的，请输出`yes`， 否则输出`no`。

开始！注意除了`yes` or `no`之外不要输出任何其他内容。
"""
