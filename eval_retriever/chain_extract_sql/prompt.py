from langchain.prompts import PromptTemplate

TEMPLATE = """I am a test_data analysis expert, familiar with common tools such as pandas, numpy, scipy, sklearn, matplotlib, pymysql, etc. 
My job is to extract the table names and field names used from the python code and retain their mapping relationships.

Example:

SQL: ```
SELECT e.name AS EmployeeName, d.department_name AS DepartmentName
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
ORDER BY d.department_name ASC
LIMIT 10;
```
Answer: tables is: ['employees', 'departments'];columns is: ['employees.name','employees.department_id','departments.department_name','departments.department_id','departments.department_name']

Begin. 
Answer in format.

SQL: ```
{code}
```
Answer: """

prompt_template = PromptTemplate(
    input_variables=["code"],
    template=TEMPLATE,
)
