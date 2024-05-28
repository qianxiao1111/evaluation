# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/28 14:10
@Auth ： zhaliangyu
@File ：tt.py
@IDE ：PyCharm
"""
import copy
import json
with open("test_samples.json", "r") as f:
    samples = json.load(f)

new_samples = []
for sample in samples:
    new_sample = copy.deepcopy(sample)
    new_sample["table_paths"] = [sample["table_paths"]]
    new_samples.append(new_sample)

with open("test_samples.json", "w") as f:
    json.dump(new_samples, f, ensure_ascii=False)


