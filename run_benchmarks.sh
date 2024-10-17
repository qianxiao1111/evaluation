export CUDA_VISIBLE_DEVICES=0

MODEL_PATHS=("/data4/models/table_llms/TableLLM-CodeQwen-7B") # 指定需要运行测评的权重

for MODEL_PATH in "${MODEL_PATHS[@]}"
do
    echo "Running scripts for model at ${MODEL_PATH}"

    # table_instruct
    python table_related_benchmarks/run_table_instruct_eval.py --model-path ${MODEL_PATH} 
    wait

    # table_bench
    python table_related_benchmarks/run_table_bench_eval.py --model_path ${MODEL_PATH}
    wait

    # reject
    python table_related_benchmarks/run_reject_eval.py --model_path ${MODEL_PATH}
    wait

    # recall
    python table_related_benchmarks/run_recall_eval.py --model_path ${MODEL_PATH}
    wait

    # nl2sql
    python table_related_benchmarks/run_text2sql_eval.py --model_path ${MODEL_PATH}
    wait 

    # MBPP
    python general_benchmarks/MBPP/eval_instruct_vllm.py --model_path ${MODEL_PATH}
    wait

    # human-eval
    python general_benchmarks/HumanEval/eval_instruct_vllm.py --model_path ${MODEL_PATH}
    wait

    # cmmlu
    python general_benchmarks/MMLU/evaluator.py --task cmmlu --lang zh --model_path ${MODEL_PATH}

    # mmlu
    python general_benchmarks/MMLU/evaluator.py --task mmlu --lang en --model_path ${MODEL_PATH}
done

echo "All models processed."
