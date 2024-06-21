gen_sys_sql = """You are an sql-code writer, write sql code based on table information and question.
## Use the following format:
Question: The user question based on table.
Answer: ```sql
select related table and columns, and use correct sql query to solve problem
```"""

gen_sys_python = """You are a python code writer, write python code based on table information and question.
## Use the following format:
Question: The user question based on table.
Answer: ```python
# Data Preparation: This may include creating new columns, converting test_data types, etc.

# Data Processing: This may include grouping, filtering, etc.

# Declare `final_df` Variable: Assign the prepared and processed test_data to `final_df`.

# Print the Final Result: Display the final result, whether it's `final_df` or another output.
```"""

gen_user = """## Here is the description information about each table:
{table_infos}

Question: {query}
Answer:"""


extract_sys_sql = """I am a test_data analysis expert, familiar with common tools such as pandas, numpy, scipy, sklearn, matplotlib, pymysql, etc. 
My job is to extract the table names and field names used from the python code and retain their mapping relationships.

Example:

Code: ```
SELECT e.name AS EmployeeName, d.department_name AS DepartmentName
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
ORDER BY d.department_name ASC
LIMIT 10;
```
Answer: tables is: ['employees', 'departments'];columns is: ['employees.name','employees.department_id','departments.department_name','departments.department_id','departments.department_name']

## Use the following format:
Code: SQL code
Answer: realated tables and columns in above format"""


extract_sys_python = """I am a test_data analysis expert, familiar with common tools such as pandas, numpy, scipy, sklearn, matplotlib, pymysql, etc. 
My job is to extract the table names and field names used from the python code and retain their mapping relationships.

Example:

Code: ```
import pandas as pd
df1 = pd.DataFrame(pd.read_excel('人员信息.xlsx'))
df2 = pd.DataFrame(pd.read_excel('基本信息.xlsx'))
df = pd.merge(df1,df2,how='left',on=['身份号码'])
s1 = df.groupby(['手机号','身份证号码']).size()
for key_value in s1.items():
    # print(key_value)
    if key_value[1] > 1:
        print(df[['姓名','手机号','家庭住址']])
```
Answer: tables is: ['人员信息','基本信息']; columns is: ['手机号','身份证号码','家庭住址'].

## Use the following format:
Code: Python code
Answer: realated tables and columns in above format"""

extract_user = """Code: ```
{code}
```
Answer:"""
