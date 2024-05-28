from langchain.prompts import PromptTemplate

TEMPLATE = """I am a test_data analysis expert, familiar with common tools such as pandas, numpy, scipy, sklearn, matplotlib, pymysql, etc. 
My job is to extract the table names and field names used from the python code and retain their mapping relationships.

Here are some examples:

SQL: ```
SELECT e.name AS EmployeeName, d.department_name AS DepartmentName
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
ORDER BY d.department_name ASC
LIMIT 10;
```
Answer: Let’s think step by step.
the tables is [employees, departments].
the columns is [employees.name,employees.department_id,departments.department_name,departments.department_id,departments.department_name]

End of examples. 

Begin.

SQL: ```
{code}
```
Answer: Let’s think step by step. """

prompt_template = PromptTemplate(
    input_variables=["code"],
    template=TEMPLATE,
)
