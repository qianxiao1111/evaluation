export CUDA_VISIBLE_DEVICES=0

python -m gen_baseline_tableinstruct \
    --model-path /data1/workspace/lly/liliyao/models/Qwen/Qwen2-7B-Instruct \
    --json-path /data4/workspace/lly/table_instruct_eval/data/TableInstruct/eval_data \
    --output-path /data4/workspace/lly/table_instruct_eval/eval/all/1-vLLM \
    --num-gpus-total 1 \
    --num-gpus-per-model 1 \
    --dataset-part all_test \
    --inference-type vLLM \
    --inference-config vLLM_config.json

wait

python -m gen_baseline_tableinstruct \
    --model-path /data1/workspace/lly/liliyao/models/Qwen/Qwen2-7B-Instruct \
    --json-path /data4/workspace/lly/table_instruct_eval/data/TableInstruct/eval_data \
    --output-path /data4/workspace/lly/table_instruct_eval/eval/all/1-TGI \
    --num-gpus-total 1 \
    --num-gpus-per-model 1 \
    --dataset-part all_test \
    --inference-type TGI \
    --inference-config TGI_config.json