# evaluation

## 1.概览

目前提供llm 三种不同能力的指标测试

1）代码纠错能力 ——code_correction

2）模糊回答拒绝能力——reject

3）多表场景选表选字段能力——retrievar

## 2.code_correction_eval

### 2.1运行脚本

```python
./evaluate_code_correction/run_eval.py
```

脚本运行分为两步：

1）生成待评价llm的code_correction 结果， 保存为results.json,默认保存在

`./evalset/code_correction_test/results.json`

2）由生成的results.json，计算pass-rate指标（目前支持execute-pass-rate和llm-eval-pass-rate两种）

### 2.2运行方法

1.修改evaluate_code_correction/run_eval.py脚本中的eval_dataset_path为对应的eval-dataset

```python
./evalset/code_correction_test/correction_set_new.json
```

2.在evaluate_code_correction/llms.py中修改llm_for_eval和llm_judge对应的配置，对于本地模型，配置可参考llm_gen

3.运行run_eval.py(备注： gen_answers()和 run_eval()函数可分别运行)

## 3.reject-eval

### 运行方法

1）运行 run_reject_eval.py

评价测试集为：

```python
evalset/reject_test/test_query.json # queries
evalset/reject_test/ground_truth.json # ground_truth
```

## 4.retrieval-eval(table_column_select)
```bash
cd eval_retriever

# 以跑10行数据为例
python run_table_select_eval.py \
    --model_path /home/dev/weights/CodeQwen1.5-7B-Chat \
    --temperature 0 \
    --max_len 8192 \
    --temperature 0.01 \
    --eval_dataset_path evalset \
    --eval_results_save_path evalset \
    --num 10
```