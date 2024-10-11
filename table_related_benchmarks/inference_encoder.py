import pandas as pd
from vllm import LLM
from vllm.sampling_params import SamplingParams
import copy


def extract_contrastive_table(df: pd.DataFrame):
    # Convert DataFrame to the desired format
    return {
        "columns": [
          {
              "name": col,
              "dtype": str(df[col].dtype),
              "contains_nan": df[col].isnull().any(),
              "is_unique":df[col].nunique() == len(df[col]),
              "values": df[col].tolist(),  # slice?
          }
          for col in df.columns
      ]
    }

import contextlib
import gc
import torch
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.utils import is_cpu


def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()

def inference_with_encoder(args, format_msg_datas):
    print("Load model...")
    model = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        # gpu_memory_utilization=0.9,
        # max_num_seqs=16,
        dtype="half",
        # dtype="bfloat16",
    )
    sparams = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    # 单个推理查看prompt
    # res = model.chat(messages=format_msg_datas[0], sampling_params=sparams)
    # print(res)
    # print("------------------PROMPT Start----------------")
    # print(res[0].prompt)
    # print("------------------PROMPT END-----------------")


    # print("++++++++++++++++++++++++Response Start++++++++++++++++++++++++")
    # print(res[0].outputs[0].text)
    # print("++++++++++++++++++++++++Response End++++++++++++++++++++++++")
    # print("Generating answers finished..")
    model_outputs = model.batch_chat(messages=format_msg_datas, sampling_params=sparams)
    model_outputs_text = [mot.outputs[0].text for mot in model_outputs]
    del model
    cleanup()
    return model_outputs_text

# format_inputs

def truncate(value, max_length=80):
    new_value = ""
    if not isinstance(value, str) or len(value) <= max_length:
        new_value = value
    else:
        new_value = value[:max_length] + "..."
    return new_value

def format_encoder_tables(df_names, table_paths):
    tables = []
    tables_info = []
    for idx, table_path in enumerate(table_paths):
        df_name = df_names[idx]
        df = pd.read_csv(table_path, encoding="utf-8", nrows=500)
        df_extra_info = extract_contrastive_table(df)
        tables_info.append(copy.deepcopy(f"Details about the '{df_name}' other info as follows:\n<TABLE_CONTENT>\n"))
        tables.append(copy.deepcopy(df_extra_info))

    return tables, tables_info

def read_df_head(table_path, head_num=3, format_type="string"):
    df = pd.read_csv(table_path, encoding="utf-8", nrows=500)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df_head = copy.deepcopy(df.head(head_num))
    df_truncated_head = df_head.apply(lambda x: x.map(lambda y: truncate(y, 80)))
    if format_type == "string":
        df_truncated_head_str = df_truncated_head.to_string()
    elif format_type == "md":
        df_truncated_head_str = df_truncated_head.to_markdown(index=False)
    else:
        df_truncated_head_str = df_truncated_head.to_string()
    return df_truncated_head_str, df

# build_message # def build_single_messages(test_dt)
# format_inputs # def format_inputs(test_datas)
