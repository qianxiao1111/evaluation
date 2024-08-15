MODEL_NAME_OR_PATH="/data3/models/DeepSeek/deepseek-coder-6.7b-base"
DATASET_ROOT="HumanEval/data"
LANGUAGE="python"
CUDA_VISIBLE_DEVICES=5,6,7 python -m accelerate.commands.launch --config_file HumanEval/test_config.yaml HumanEval/eval_pal.py --model_path ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}