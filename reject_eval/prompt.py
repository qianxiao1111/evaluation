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

output_content_classify_system = """
You need to determine whether the meaning of the input sentence is positive, negative, or uncertain.

## Use the following format:
Sentence: The input sentence you need to judge
Answer: Directly give a judgment of whether the sentence, and give a 'positive' or 'negative' or 'uncertain' response.

## Examples:

Example 1:
Sentence: The question can be answered.
Answer: positive

Example 2:
Sentence: Based on the provided dataframe information, the question cannot be answered. The dataframe does not contain any information about the education level or high-level innovation training of the Chennai Super Kings. Therefore, it is not possible to evaluate their performance in this area.
Answer: negative

Example 3:
Sentence: I don't know or I am not sure.
Answer: uncertain
"""

output_content_classify_instruct = """
Sentence: {input}
Answer: 
"""


