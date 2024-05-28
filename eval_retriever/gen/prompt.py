from langchain.prompts import PromptTemplate

TEMPLATE = """You are an sql-code writer, write sql code based on below information:
{table_infos}

Question:
{query}


follow the plan write the sql query 

sql:
```sql
select related table and columns, and use correct sql query to solve problem
```

Answer in format:"""
prompt_template = PromptTemplate(
    input_variables=["table_infos", "query"],
    template=TEMPLATE,
)


TEMPLATE_PY = """You are a python code writer, write python code based on below information:
dfs:
{table_infos}

Question:
{query}


plan to solve the question step by step:
<format>
plan:
1) related table and columns

2) how to calc the result

3) check is drop null value、 deduplicate necessary

</format>

follow the plan write python
surpose df is already loaded
<format>

```python
import pandas as pd

# Data Preparation: This may include creating new columns, converting test_data types, etc.

# Data Processing: This may include grouping, filtering, etc.

# Declare `final_df` Variable: Assign the prepared and processed test_data to `final_df`.

# Print the Final Result: Display the final result, whether it's `final_df` or another output.
```
</format>```

Answer in Format:"""
prompt_template_py = PromptTemplate(
    input_variables=["table_infos", "query"],
    template=TEMPLATE_PY,
)

# {{Provide the Python code here to solve the problem}}
