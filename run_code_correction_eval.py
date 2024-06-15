"""
@Time ： 2024/6/15 09:52
@Auth ： zhaliangyu
@File ：run_code_correction_eval.py
@IDE ：PyCharm
"""

import time
from util import start_service, is_service_up
from langchain_openai import ChatOpenAI
from evaluate_code_correction.run_eval import get_results, run_eval

def get_extra_infer_kwargs(args) -> dict:
    """llm top-k top-p kwargs"""
    top_k = args.top_k  if args.top_k else None
    top_p = args.top_p if args.top_p else None
    kwargs = {"top_k": top_k, "top_p": top_p}
    return kwargs

def main(args):
    """main function to run the code correction evaluation"""
    model_path = args.model_path
    cutoff_len = args.cutoff_len
    k = args.k
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature if args.temperature else 1.0
    eval_dataset_path = args.eval_dataset_path
    eval_results_save_path = args.eval_results_save_path
    if args.run_llm_eval:
        from evaluate_code_correction.llms import llm_judge
        llm_for_judge = llm_judge
    else:
        llm_for_judge = None
    model_kwargs = {}
    # 启动vllm 模型服务
    service_process, port, model_name = start_service(model_path, cutoff_len)
    # 等待服务启动
    service_url = f"http://localhost:{port}"
    while not is_service_up(service_url):
        print("Waiting for the service to start...")
        time.sleep(3)
    time.sleep(2)
    print("服务已启动")
    # 业务代码
    service_openai_url = service_url + "/v1"
    # 初始化评估LLM
    llm_eval = ChatOpenAI(
        temperature=temperature,
        openai_api_base=service_openai_url,
        openai_api_key="none",
        model_name=model_name,
        max_tokens=max_new_tokens,
        model_kwargs=model_kwargs
    )

    # Eval-results 生成
    # -----------------------------
    get_results(eval_dataset_path = eval_dataset_path,
                llm_for_eval=llm_eval,
                result_path=eval_results_save_path,
                k=k)
    # -----------------------------

    # Eval 指标计算
    # -----------------------------
    run_eval(eval_result_path=eval_results_save_path,
             llm_for_judge=llm_for_judge)
    # -----------------------------

    # 关闭模型服务
    service_process.terminate()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="eval code_correction")
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature setting')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of output tokens')
    parser.add_argument('--cutoff_len', type=int, default=8192, help='Cutoff length')
    parser.add_argument('--eval_dataset_path', type=str, default="evalset/code_correction_test/correction_set_new.json", help='Test Set Path')
    parser.add_argument('--k', type=int, default=1, help='Max iteration for llm to run each code correction task')
    parser.add_argument('--eval_results_save_path', type=str, default="evalset/code_correction_test/results.json", help='Max iteration for llm to run each code correction task')
    parser.add_argument('--run_llm_eval', type=bool, default=False, help='Whether use another llm to judge the eval-results, if set to `True`, modify the `evaluate_code_correction/llms.py` configs')
    args = parser.parse_args()
    main(args)

