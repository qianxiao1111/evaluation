import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from humaneval import HumanEval as evaltor
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./output")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data3/models/DeepSeek/deepseek-coder-6.7b-base",
    )
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--dataroot", type=str, default="HumanEval/data")
    args = parser.parse_args()

    logdir = args.logdir
    language = args.language
    model_path = args.model_path

    if logdir == "":
        logdir = "tmp/"
    tokenizer = dict(
        cls=AutoTokenizer,
        model_path=model_path,
    )

    dataroot = args.dataroot

    evaluator = evaltor(
        data_root=dataroot,
        max_seq_len=4096,
        tokenizer_cfg=tokenizer,
        log_dir=logdir,
        n_sample=1,
        batch_size=1,
        language=language,
        max_gen_len=500,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=accelerator.device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluator.eval_model(model, accelerator)
