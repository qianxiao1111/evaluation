eval_system = """
You are a data scientist. Based on the provided dataframes information, determine if the question can be answered.

## Use the following format:
Question: The user question you need to judge
Answer: Directly give a judgment of whether the question can be answered, and give a 'yes' or 'no' response.

## Examples:

Example 1:
Question: Can you find the average sales for the last quarter from table?
Answer: yes

Example 2:
Question: Can you determine the impact of marketing campaigns on sales for the last five years using the table?
Answer: no
"""

eval_instruction = """
## Only use the following tables::
{df_info}

Question: {input}
Answer: """

