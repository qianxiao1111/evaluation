from util import start_service, is_service_up
from reject_eval.run_eval import run_eval
import argparse
import time

def main(args):
    temperature = args.temperature
    model_path = args.model_path
    max_tokens = args.max_tokens
    cutoff_len = args.cutoff_len
    test_path = args.test_path

    # 启动服务
    service_process, port, model_name = start_service(model_path, cutoff_len)

    service_url = f"http://localhost:{port}"
    while not is_service_up(service_url):
        print("Waiting for the service to start...")
        time.sleep(1)
    time.sleep(2)
    service_openai_url = service_url + "/v1"
    print("服务已启动！")

    # eval 评估
    # -----------------------------
    run_eval(test_path, model_name, temperature, max_tokens, service_openai_url)
    # -----------------------------

    # 关闭服务
    service_process.terminate()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval reject")
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature setting')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum number of output tokens')
    parser.add_argument('--cutoff_len', type=int, default=8192, help='Cutoff length')
    parser.add_argument('--test_path', type=str, default="evalset/reject_test/test_query.json", help='Test File Path')

    args = parser.parse_args()
    main(args)


# example /home/dev/weights/CodeQwen1.5-7B-Chat
"""
python run_reject_eval.py \
    --model_path /data0/pretrained-models/checkpoints/qwen2/checkpoint-1200 \
    --temperature 0 \
    --cutoff_len 16384 \
    --max_tokens 1024
"""
