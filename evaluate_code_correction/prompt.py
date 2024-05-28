# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/25 15:52
@Auth ： zhaliangyu
@File ：prompt.py
@IDE ：PyCharm
"""

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
Observation: 动作输出的观察内容
... (这个 Thought/Python Code/Observation 过程可以重复N次)
Thought: 根据观察的结果做出总结, Final Answer:
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
当前时间: {current_time}
{agent_scratchpad}
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

PROMPT_PYTHON_GENERATE_NORMAL = """
你正在使用Python代码处理一个命名为`df`的pandas分析任务。
以下是输入表格的信息:
{df_head}

你需要严格遵循以下格式来生成内容：

<format>
Question: 需要解决的问题，可以对问题进行适度改写方便生成步骤和python code
Thought: 思考解决问题的步骤
Python Code:
```python
# Data preparation: 这一步可能包括创建新列、转换数据类型等。

# Data processing: 这一步可能包括分组、过滤等。

# Declare `final_df` var: 将经过准备和处理后的数据分配给`final_df`。

# Print the final result based on the question: 打印符合题意的最终结果
```
</format>

开始! 以规定的格式(Thought/Python Code)回答问题。
当前时间: {current_time}
<format>
Question: {input}
"""

PROMPT_PYTHON_GENERATE_VISUAL = """
你正在使用Python代码处理一个命名为`df`的pandas分析任务。 分析的过程包含可视化的内容。
以下是输入表格的信息:
{df_head}

你需要严格遵循以下格式来生成内容：

<format>
Question: 需要解决的问题，可以对问题进行适度改写方便生成步骤和python code
Thought: 思考解决问题的步骤
Python Code:
```python
# Data preparation: 这一步可能包括创建新列、转换数据类型等。

# Data processing: 这一步可能包括分组、过滤等。

# Declare `final_df` var: 将经过准备和处理后的数据分配给`final_df`。

# Visualization:  使用代码来绘制图表，请添加标题、标签和图例。

# Print the final result based on the question: 打印符合题意的最终结果
```
</format>

开始! 以规定的格式(Thought/Python Code)回答问题。
当前时间: {current_time}
<format>
Question: {input}
"""



