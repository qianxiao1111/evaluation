set -ex

export CUDA_VISIBLE_DEVICES="1"
MODEL_NAME_OR_PATH="/data4/sft_output/qwen2.5-ins-1012/checkpoint-3200"

PROMPT_TYPE="qwen25-math-cot"
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="gsm8k,math"
TOKENIZERS_PARALLELISM=false \
python3 -u general_benchmarks/math/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite