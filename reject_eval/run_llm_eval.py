from prompt import eval_system, eval_instruction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
import sys
import os

from eval_metrics import load_json, save_json, evaluation


def build_chain():
    os.environ['https_proxy'] = 'http://10.0.0.46:7979'
    llm = ChatOpenAI(
        # openai_api_base="https://api.openai.com",
        openai_api_key="sk-proj-vdxxxNg",
        model_name="gpt-3.5-turbo-0125",
        # max_tokens=1024,
        max_tokens=1536,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", eval_system),
            ("human", eval_instruction),
        ]
    )
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    return chat_llm_chain


def llm_eval(test_file_path):
    # 加载问题
    test_datas = load_json(test_file_path)
    processed_data = []
    for test_dt in test_datas:
        query = test_dt["query"]
        df_info_str = test_dt["df_info"]
        
        chat_llm_chain = build_chain()
        res = chat_llm_chain.invoke({
            "df_info": df_info_str,
            "input": query,
        })
        llm_output = res["text"]
        # 解析
        if "yes" in llm_output.lower():
            test_dt["is_reject"] = False
        elif "no" in llm_output.lower():
            test_dt["is_reject"] = True
        else:
            print("解析错误")
        
        print(llm_output)
        processed_data.append(test_dt)
    
    save_json("test_data/llm_output_data.json", processed_data)
    evaluation("test_data/ground_truth.json", "test_data/llm_output_data.json")


if __name__ == "__main__":
    test_file_path = "test_data/test_query.json"
    llm_eval(test_file_path)