# Benchmarks evaluations for tablegpt

<p align="center">
    <a href="#-About">🔥About</a> •
    <a href="#-Usage">💻Usage</a> •
</p>

## About

</div>

This is a repo opened for evaluation on different table-related benchmarks for tablegpt. 

Given the complexity of table QA tasks and the uncertainty of input instructions,  we provide evaluation datasets and scripts for 7 capabilities: 

- ✨Code correction based on tables 
- ✨Refusal of ambiguous questions
- ✨Table & field recall in multi-table scenarios
- ✨Table QA output code executable
- ✨Table-Bench.
- ✨Text2Sql.
- ✨TableInstruct, which includes a series of table-related evaluation benchmarks.

In addition, we have integrated other general abilities benchmarks like HumanEval, MBPP and MMLU/CMMLU.
We have built an inference method based on the  local model path using vLLM as the backend, and defined a set of example prompts templates for the above benchmarks. 

## Usage

</div>
</details>

⏬ To use this framework, please first install the repository from GitHub:

```shell
git clone https://github.com/tablegpt/tablegpt-eval
cd tablegpt-eval
pip install -r requirements.txt
```

</div>
</details>

[!Tips]
1. You can run all the benchmarks with the default params by running command `bash run_benchmarks.sh`.
2. If you want more configuration options for running parameters, refer to the typical Python script.
3. Download the .db files before running text2sql evaluation scripts. Download urls refer to `/table_related_benchmarks/evalset/bird_data/dev_databases/README.md`(Bird dev) & `table_related_benchmarks/evalset/spider_data/dev_database/README.md` (Spider dev) & `table_related_benchmarks/evalset/spider_data/test_database/README.md` (Spider test).


### Code correction eval

We provide a non-executable eval dataset based on the Python language. Eval dataset path:

```python
table_related_benchmarks/evalset/code_correction_test/correction_set.json
```

We use the  ***executable_pass_rate*** and ***absolute_match_rate***  of the corrected code in pass-1 to evaluate the model's code correction ability. You can perform code-correction evaluation by running the following Python command:

```bash
python table_related_benchmarks/run_code_correction_eval.py \
		--model_path  <EVAL MODEL PATH> \
		--template  <CHAT_TEMPLATE_NAME, support [llama3, baichuan, chatglm, None], default None> \
    	--eval_results_save_path <PATH TO SAVE THE EVAL RESULTS> \
        --gpus_num <NUMBER OF GPU TO RUN INFERENCE> \
        --temperature <ONE OF THE INFERENCE PARAMETER>
```

### Ambiguous reject eval

We provide 298 table-based queries, with a ratio of about 1:3 between queries marked as ambiguous (to be rejected) and queries that should be accepted and correctly answered. Dataset path:

```python
# test queries
evalset/reject_test/test_query.json
# queries with ground truth
evalset/reject_test/ground_truth.json
```

We use **accuracy**, **recall**, and **F1 score** as metrics to evaluate the LLM's ability in this task. You can perform reject evaluation by  running the following Python command:

```bash
python table_related_benchmarks/run_reject_eval.py \
    --model_path <EVAL MODEL PATH>  \
    --save_path <LLM OUTPUT CONTENT SAVE PATH> \
    --gpus_num <NUMBER OF GPU TO RUN INFERENCE> \
    --temperature <ONE OF THE INFERENCE PARAMETER>
```

### Table&Fields recall eval

The provided eval dataset path:

```python
table_related_benchmarks/evalset/retrieval_test/recall_set.json
```

We use a series of evaluation metrics such as **recall**, **precision**, **Jaccard similarity**, and **Hamming loss** to assess the LLM's performance in table and field retrieval tasks.  You can perform recall evaluation by  running the following Python command:

```bash
python table_related_benchmarks/run_recall_eval.py \
    --model_path <EVAL MODEL PATH> \
    --temperature <TEMPERATURE> \
    --gpus_num <NUMBER OF GPU TO RUN INFERENCE> 
```

### Table QA executable 

Provide 2178 table based queries,  eval dataset path:

```python
table_related_benchmarks/evalset/table_qa_execuate_test/tableqa_samples_with_paths.jsonl
```

We employ ***executable_pass_rate***  of pass-1 to employ the model's tableQA code generation ability. You can perform tableQA evaluation by  running the following Python command:

```bash
python table_related_benchmarks/run_tableqa_execution_eval.py \
    --model_path <EVAL MODEL PATH> \
    --temperature <ONE OF THE INFERENCE PARAMETER> \
    --gpus_num <NUMBER OF GPU TO RUN INFERENCE> 
```

### TableBench evaluation

The provided eval dataset path:

```python
table_related_benchmarks/evalset/TableBench
```

In the evaluation of TableBench, Rough-L was used to assess general QA questions, while pass@1 was used as the evaluation metric for visualization-type samples.  You can perform TableBench evaluation by the following command:

```bash
python table_related_benchmarks/run_table_bench_eval.py \
    --model_path <EVAL MODEL PATH> \
    --temperature <ONE OF THE INFERENCE PARAMETER> \
    --gpus_num <NUMBER OF GPU TO RUN INFERENCE> 
```

### TableInstruct

The provided eval dataset path:

```python
table_related_benchmarks/evalset/TableInstruct
```

You can perform TableInstruct evaluation by the following command:

```bash
python table_related_benchmarks/run_table_instruct_eval.py \
    --model_path <EVAL MODEL PATH> \
    --temperature <ONE OF THE INFERENCE PARAMETER> \
    --gpus_num <NUMBER OF GPU TO RUN INFERENCE> 
```

### Text2Sql
```bash
python table_related_benchmarks/run_text2sql_eval.py --model_path <EVAL MODEL PATH> 
```

### HumanEval
Perform HumanEval evaluation by the following command:

```bash
python general_benchmarks/HumanEval/eval_instruct_vllm.py --model_path <EVAL MODEL PATH> 
```

### MBPP
Perform MBPP evaluation by the following command:

```bash
python general_benchmarks/MBPP/eval_instruct_vllm.py --model_path <EVAL MODEL PATH> 
```

### MMLU & CMMLU

```bash
python general_benchmarks/MMLU/evaluator.py --task <mmlu or cmmlu> --lang <en or zh> --model_path <EVAL MODEL PATH> 
```

