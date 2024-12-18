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
from mbpp import MBPP as evaltor
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--dataroot", type=str, default="")
    args = parser.parse_args()

    logdir = args.logdir

    if logdir == "":
        logdir = "tmp/"
    tokenizer = dict(
        cls=AutoTokenizer,
        model_path=logdir,
    )

    dataroot = args.dataroot

    evaluator = evaltor(
        data_root=dataroot,
        max_seq_len=4096,
        tokenizer_cfg=tokenizer,
        log_dir=logdir,
        n_sample=1,
        batch_size=1,
        max_gen_len=500,
    )
    model = AutoModelForCausalLM.from_pretrained(
        logdir,
        device_map=accelerator.device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluator.eval_model(model, accelerator)
