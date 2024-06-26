# Table-llm-eval： An Open-Source tabular data related tasks evaluation framework

<p align="center">
    <a href="#-About">🔥About</a> •
    <a href="#-Usage">💻Usage</a> •
</p>

## About

</div>

Table-llm-eval is a project designed to support the evaluation of large language model (LLM) capabilities related to table data. 

Given the complexity of table QA tasks and the uncertainty of input instructions, unlike typical table-based QA tasks,  we provide evaluation datasets and scripts for three capabilities: 

- ✨Code correction based on tables 
- ✨Refusal of ambiguous questions
- ✨Table & field recall in multi-table scenarios.

We have built an inference method based on the local model path using VLLM as the backend, and defined a set of example prompts templates for the three tasks: code correction, ambiguous question refusal, and table and field recall. 	You also can define your own prompt templates. 

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

[!Tip]

If you want more configuration options for running parameters, refer to the typical Python script.

### Code correction eval

We provide a non-executable eval dataset based on the Python language. Eval dataset path:

```python
evalset/code_correction_test/correction_set.json
```

We use the  ***executable_pass_rate***  of the corrected code in pass-1 to evaluate the model's code correction ability. You can perform code-correction evaluation by running the following Python command:

```bash
python run_code_correction_eval.py \
		--model_path  <EVAL MODEL PATH> \
		--template  <CHAT_TEMPLATE_NAME, support [llama3, baichuan, chatglm, None], default None> \
    	--eval_results_save_path <PATH TO SAVE THE EVAL RESULTS> \
        --gpus_num <NUMBER OF GPU TO RUN INFERENCE> \
        --temperature <ONE OF THE INFERENCE PARAMETER>
```

### Ambiguous reject eval

We provide 300 table-based queries, with a ratio of 1:3 between queries marked as ambiguous (to be rejected) and queries that should be accepted and correctly answered. Dataset path:

```python
# test queries
evalset/reject_test/test_query.json
# queries with ground truth
evalset/reject_test/ground_truth.json
```

We use **accuracy**, **recall**, and **F1 score** as metrics to evaluate the LLM's ability in this task. You can perform reject evaluation by  running the following Python command:

```bash
python run_reject_eval.py \
    --model_path <EVAL MODEL PATH>  \
    --save_path <LLM OUTPUT CONTENT SAVE PATH> \
    --gpus_num <NUMBER OF GPU TO RUN INFERENCE> \
    --temperature <ONE OF THE INFERENCE PARAMETER>
```

### Table&Fields recall eval

The provided eval dataset path:

```python
evalset/retrieval_test/recall_set.json
```

We use a series of evaluation metrics such as **recall**, **precision**, **Jaccard similarity**, and **Hamming loss** to assess the LLM's performance in table and field retrieval tasks.  You can perform recall evaluation by  running the following Python command:

```bash
python run_recall_eval.py \
    --model_path <EVAL MODEL PATH> \
    --temperature <LLM OUTPUT CONTENT SAVE PATH> \
    --gpus_num <NUMBER OF GPU TO RUN INFERENCE> 
```







## 