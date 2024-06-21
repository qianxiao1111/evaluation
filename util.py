# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/14 17:15
@Auth ： zhaliangyu
@File ：util.py
@IDE ：PyCharm
"""
import json

def load_json(data_path):
    """
    # 加载 json 文件
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data_path, data_list):
    """
    # 保存 json 文件
    """
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False)