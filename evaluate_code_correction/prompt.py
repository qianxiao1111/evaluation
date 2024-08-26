# -*- coding: utf-8 -*-

# RECTIFY_PROMPT_PYTHON_SYSTEM = """
# 你现在正在充当一名数据分析任务的Python代码审查员，并能访问对应的pandas Dataframes。

# 你需要根据输入的查询、表格信息以及运行时错误信息，对输入的原始代码和代码思路进行修改，以获得正确的运行结果。

# 请输出以下格式的内容：

# Thought: 解释错误的原因并提供正确的解决方法。

# Python Code:
# ```python
# # 数据预处理: 如果需要，对数据进行预处理和清理，避免直接使用 `pd.DataFrame` 进行分析数据的获取。

# # 数据分析: 对数据进行分析操作，例如分组、过滤、聚合等。

# # 声明 `final_df` 变量: 将数据准备和处理的结果赋值给 `final_df`。

# # 根据问题打印最终结果
# ```
# """

RECTIFY_PROMPT_PYTHON_SYSTEM ="""
You are now acting as a Python code reviewer for data analysis tasks and have access to the corresponding pandas DataFrames.

Based on the input query, table information, and runtime error messages, you need to modify the provided original code and thought process to achieve the correct result.

Please output the content in the following format:

Thought: Explain the cause of the error and provide the correct solution.
Python Code:
```python
# Data Preprocessing: If necessary, preprocess and clean the data. Avoid using `pd.DataFrame` to obtain analysis data.

# Data Analysis: Perform data analysis operations such as grouping, filtering, aggregating, etc.

# Declare `final_df` Variable: Assign the result of data preparation and processing to `final_df`.

# Print the final result based on the query
```
"""

# RECTIFY_PROMPT_PYTHON_INSTRUCTION = """
# Here is the input table information:
# {table_infos}

# Here is the input query:
# {query}

# Here is the input code and thought process:
# {output}

# Here is the result of running the original Python code (Observe):
# {observe}

# Let's start!
# Please correct the input code according to the format above, ensuring the code produces the correct result based on the user's input and table information. Note that you do not need to use pd.DataFrame to define analysis data.

# Current time: {current_time}
# """

RECTIFY_PROMPT_PYTHON_INSTRUCTION = """
Given the following inputs:
Table Information:
{table_infos}

Question:
{query}

Generated Output by the Model:
{output}

Execution Result of the Code:
{observe}

Current time: {current_time}

Your Task:
- Explanation: Provide a brief explanation of the error and how you corrected it.
- Correct the Error: Based on the table information, the question, the generated code, and the execution result, identify and correct the error(s) in the code. Ensure that the corrected code correctly answers the question using the provided table.
"""

CLASSIFY_PROMPT_PYTHON = """
你现在正在充当一名Python代码reviewer，输入思考过程和代码以及执行结果是根据用户的query和表格信息生成的。
但由于程序员水平有限，这个输入的代码可能是错的。 你需要根据用户的query、代码的执行结果和真实结果，来对代码以及代码执行结果的正确性进行判断。

以下是输入的query
{query}

以下是程序员根据query和表格信息所生成的思考过程和代码
{code}

以下是代码的执行观测结果：
{observation}

以下是用户问题对应的真实结果：
{true_result}

请根据以上内容，判定代码的正确性。
如果认为代码是正确的，请输出`yes`， 否则输出`no`。

开始！注意除了`yes` or `no`之外不要输出任何其他内容。
"""





