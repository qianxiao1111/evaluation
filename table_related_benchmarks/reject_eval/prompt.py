eval_system = """
You have access to a number of pandas dataframes. Given a user question about the dataframes, write the Python code to answer it.
if the given query is not clear or does not relate to the given dataframes, just reply `no`.

## Use the following format:
Question: The user's query.
Thought: Analyze the provided dataframes and the question to determine the solution.
Python Code: Generate the Python code, or you can just reply `no`.
"""

eval_instruction = """
## Information on each dataframe is given below:

{df_info}

Question: {input}
"""


# eval_system = """
# You are a data scientist. Based on the provided dataframes information, determine if the question can be answered.

# ## Use the following format:
# Question: The user question you need to judge
# Answer: Directly give a judgment of whether the question can be answered, and give a 'yes' or 'no' response.

# ## Examples:

# Example 1:
# Question: Can you find the average sales for the last quarter from table?
# Answer: yes

# Example 2:
# Question: Can you determine the impact of marketing campaigns on sales for the last five years using the table?
# Answer: no
# """

# eval_instruction = """
# ## Only use the following tables::
# {df_info}

# Question: {input}
# Answer: """

output_content_classify_system = """
You need to determine whether the meaning of the input context is positive, negative, or uncertain.

## Use the following format:
Context: The input context you need to judge
Answer: positive, negative, or uncertain

## Examples:

Example 1:
Context: The question can be answered.
Answer: positive

Example 2:
Context: Based on the provided dataframe information, the question cannot be answered. The dataframe does not contain any information about the education level or high-level innovation training of the Chennai Super Kings. Therefore, it is not possible to evaluate their performance in this area.
Answer: negative

Example 3:
Context: I don't know or I am not sure.
Answer: uncertain
"""

output_content_classify_instruct = """
## This is a new task, directly give a judgment the sentence, and output a 'positive' or 'negative' or 'uncertain' response without any other content.
Context: {input}
Answer: 
"""
