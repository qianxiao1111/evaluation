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

TEMPLATE_REGEN = """Acting as a Python code reviewer, your task is to refine the code based on the given query, table details, and error messages from its execution.

Your output should adhere to this format:

```
Thought: Outline the steps to address the issue.
Python Code:
```python
# Data Preparation: This may entail creating new columns, converting data types, etc.

# Data Processing: Steps like grouping, filtering, etc., can be applied here.

# Define `final_df`: Assign the processed dataset to `final_df`.

# Display the Final Output: Print the outcome conforming to the query's requirements.
```

Provided below is the table information:
{table_infos}

Here is the input query:
{query}

Original Thought Process
{cot}

Original Python Code:
{code}

Observations from Executing the Original Python Code:
{observation}

begin

Please revise the submitted code following the agreed-upon format above.

"""

prompt_template_regen = PromptTemplate(
    input_variables=["table_infos", "query", "cot", "code", "observation"],
    template=TEMPLATE_REGEN,
)
