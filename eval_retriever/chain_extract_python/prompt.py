from langchain.prompts import PromptTemplate

TEMPLATE = """I am a test_data analysis expert, familiar with common tools such as pandas, numpy, scipy, sklearn, matplotlib, pymysql, etc. 
My job is to extract the table names and field names used from the python code and retain their mapping relationships.

Here are some examples:

Python: ```
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
Answer: Let’s think step by step.
the tables is [人员信息.xlsx,基本信息.xlsx].
the columns is [手机号,身份证号码,家庭住址]

End of examples. 

Begin.

Python: ```
{code}
```
Answer: Let’s think step by step. """

prompt_template = PromptTemplate(
    input_variables=["code"],
    template=TEMPLATE,
)
