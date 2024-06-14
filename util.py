# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/14 17:15
@Auth ： zhaliangyu
@File ：util.py
@IDE ：PyCharm
"""
import random
import subprocess
import requests
import socket
import time

def check_port_in_use(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False  # 端口未被占用
        except socket.error as e:
            return True  # 端口被占用

def start_service(model_path, max_len):
    while True:
        port = random.randint(8000, 10000)
        if not check_port_in_use(port):
            break

    max_len_cmd = str(int(max_len))
    port_cmd = str(int(port))
    model_name = f"vllm_serve_{port}"
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", model_name,
        "--max-model-len", max_len_cmd,
        "--gpu-memory-utilization", "0.8",
        "--port", port_cmd
    ]
    process = subprocess.Popen(command)
    return process, port, model_name

# 检查服务是否启动
def is_service_up(url):
    service_openai_url = url + "/v1/models"
    try:
        response = requests.get(service_openai_url)
        return response.status_code == 200
    except requests.ConnectionError:
        return False
