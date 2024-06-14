from reject_eval.prompt import eval_system, eval_instruction
from reject_eval.eval_metrics import load_json, save_json, evaluation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.llm import LLMChain
import sys
import os


def build_chain(openai_url, model_name, max_tokens, temperature):
    # this is the eval-model
    llm = ChatOpenAI(
        openai_api_base=openai_url,
        openai_api_key="empty",
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
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
        # verbose=True
    )

    return chat_llm_chain

def run_eval(test_file_path, model_name, temperature, max_tokens, openai_url):
    # 加载问题
    test_datas = load_json(test_file_path)
    processed_data = []
    test_datas_len = len(test_datas)
    for idx, test_dt in enumerate(test_datas):
        print(f"进度：{idx}/{test_datas_len}")
        query = test_dt["query"]
        df_info_str = test_dt["df_info"]

        chat_llm_chain = build_chain(openai_url, model_name, max_tokens, temperature)
        res = chat_llm_chain.invoke({
            "df_info": df_info_str,
            "input": query,
        })
        llm_output = res["text"]
        test_dt["llm_output"] = llm_output
        # 解析
        if "yes" in llm_output.lower():
            test_dt["is_reject"] = False
        elif "no" in llm_output.lower():
            test_dt["is_reject"] = True
        else:
            print("解析错误")
        
        print(llm_output)
        processed_data.append(test_dt)

    # 保存路径
    parent_path = os.path.dirname(test_file_path)
    save_path = os.path.join(parent_path, 'llm_output_data.json')
    ground_truth_path = os.path.join(parent_path, 'ground_truth.json')

    save_json(save_path, processed_data)
    evaluation(ground_truth_path, save_path)
