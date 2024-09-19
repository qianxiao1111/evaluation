import os 
import pandas as pd 
import json
import torch

from tqdm import tqdm
from encoder_models.encoder1.format import dataframe_info_combined
from encoder_models.encoder1.build_input import build_instruction
from encoder_models.encoder2.build_input import build_encoder_input
from table_bench_eval.run_eval import run_eval, execute_samples_and_save

from table_bench_eval.utils import (
    read_json_file, 
    pre_save_table_to_csv,
    reformat_instruction
)

def build_template(sample):
    instruction = sample["instruction"]
    r_instruction = reformat_instruction(instruction)
    table = sample["table"]
    table = json.loads(table)
    pre_save_table_to_csv(table)
    csv_paths = ["table.csv"]
    df_names = ["df"]
    df_list = [pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False,
        nrows=500
    ) for path in csv_paths]
    df_info_list = [dataframe_info_combined(df, df_name) for df, df_name in zip(df_list, df_names)]
    table_infos = "\n\n".join(df_info_list)
    template = r_instruction.format(table_info=table_infos)
    return template

def format_inputs(samples, tokenizer, model_type="chat_model"):
    msgs = []
    encoder_inputs = []
    for sample in samples:
        instruction = build_template(sample)
        decoder_input = build_instruction(instruction, tokenizer)
        msgs.append(decoder_input)
        if model_type == "1":
            encoder_input = None
        else:
            table_info = sample["table"]
            encoder_input = build_encoder_input(table_info)
        encoder_inputs.append(encoder_input)
    return msgs, encoder_inputs

@torch.inference_mode()
def model_infer_and_save(
    model, 
    test_path,
    temperature,
    max_new_tokens,
    base_model_name,
    inference_output_dir,
    tokenizer,
    device: str = "cuda",
    model_type: str = "1",
    n_samples_test: int = 10
):
    model.to(device)
    fnames = [x for x in os.listdir(test_path) if x.endswith('.jsonl')]
    all_samples = []
    for file_name in fnames:
        print(file_name)
        file_path = os.path.join(test_path, file_name)
        samples = read_json_file(file_path)
        if n_samples_test:
            samples = samples[:n_samples_test]
        decoder_inputs, encoder_inputs = format_inputs(samples, tokenizer)
        # model_outputs = []
        for i, (decoder_inp, encoder_inp) in tqdm(enumerate(zip(decoder_inputs, encoder_inputs)), total=len(decoder_inputs)):
            # output = {}
            table = samples[i]["table"]
            table = json.loads(table)
            pre_save_table_to_csv(table)
            table_path = ["table.csv"]
            # df_names = ["df"]
            if model_type == "1":
                model_output = model.generate(
                    [decoder_inp],
                    max_new_tokens=max_new_tokens, 
                    eos_token_id = model.tokenizer.eos_token_id, 
                    pad_token_id = model.tokenizer.eos_token_id,
                    temperature = temperature,
                    path_csv = [table_path],
                    do_sample=True
                )
                output_content = model_output[1][0]
            elif model_type == "2":
                model_output = model.generate(
                    decoder_inp,
                    max_new_tokens=5000, 
                    eos_token_id = model.tokenizer_decoder.eos_token_id, 
                    pad_token_id=model.tokenizer_decoder.eos_token_id, 
                    table_str=encoder_inp
                )
                output_content = model.tokenizer_decoder.batch_decode(model_output, skip_special_tokens=True)[0]
            else:
                raise
                
            samples[i]["raw_generation"] = output_content

        save_path = os.path.join(inference_output_dir, base_model_name.split('/')[-1]+'_infer_'+file_name.split('.')[0]+'.jsonl')
        with open(save_path, 'w') as f:
            for item in samples:
                f.write(json.dumps(item)+'\n')
        all_samples.extend(samples)

    return all_samples

def main(args):
    patch_model_path = args.patch_model_path 
    model_type = args.model_type
    encoder_model_path = args.encoder_model_path
    decoder_model_path = args.decoder_model_path
    eval_results_save_path = args.eval_results_save_path
    eval_dataset_path = args.eval_dataset_path
    device=args.device
    num_samples = args.num_samples_to_eval
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens

    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    if model_type == "1":
        from encoder_models.encoder1.model_sft import Model
        model = Model.from_pretrained(
            path=encoder_model_path,
            sentence_transformer_path=patch_model_path,
            base_model_path=decoder_model_path
        )
        tokenizer = model.tokenizer
    elif model_type == "2":
        from encoder_models.encoder2.model_encoderdecoder_t5 import Model
        model = Model.from_pretrained(
            encoder_path=encoder_model_path,
            decoder_path=decoder_model_path,
            projector_path=patch_model_path
        ).float()
        tokenizer = model.tokenizer_decoder
    else:
        raise

    if model_type == "1":
        os.environ["SENTENCE_TRANSFORMER"] = os.path.split(patch_model_path)[-1]
        os.environ['MODELS_PATH'] = os.path.dirname(patch_model_path)
    
    all_samples = model_infer_and_save(
        model=model,
        test_path=eval_dataset_path,
        temperature=temperature,
        device=device,
        tokenizer=tokenizer,
        inference_output_dir=eval_results_save_path,
        base_model_name=encoder_model_path,
        max_new_tokens=max_new_tokens,
        n_samples_test=num_samples,
        model_type=model_type
    )

    all_samples = execute_samples_and_save(all_samples, eval_results_save_path, encoder_model_path)
    # eval and save results
    run_eval(all_samples, eval_results_save_path, encoder_model_path)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="eval encoder code correction")

    parser.add_argument(
        "--decoder_model_path",
        type=str,
        # required=True,
        default="/data4/sft_output/qwen2-base-0802/checkpoint-2400",
        help="Decoder base model path.",
    )

    parser.add_argument(
        "--patch_model_path",
        type=str,
        default="/data0/workspace/liliyao/saved_models/projector-0812/projector.bin",
        help="Patch model path, for encoder1 this path is sentence transformer path, for encoder2 this path is projector weigtht path",
    )

    parser.add_argument(
        "--encoder_model_path",
        type=str,
        default="/data0/workspace/liliyao/saved_models/checkpoint-364",
        help="Encoder model path",
    )

    parser.add_argument(
        "--model_type",
        choices=["1", "2"],
        default="2",
        help="Encoder1 or Encoder2",
    )

    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="evalset/TableBench",
        help="Test Set Path",
    )

    parser.add_argument(
        "--eval_results_save_path",
        type=str,
        default="evalset/TableBench/eval_results",
        # help="Max iteration for llm to run each code correction task",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Use cuda or cpu to run eval",
    )

    parser.add_argument(
        "--num_samples_to_eval",
        type=int,
        default=10,
        help="Set eval samples number to eval",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Temperature setting"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of output new tokens",
    )

    parser.add_argument(
        "--run_llm_eval",
        type=bool,
        default=False,
        help="Whether use another llm to judge the eval-results, if set to `True`, modify the `evaluate_code_correction/llms.py` configs",
    )

    args = parser.parse_args()
    main(args)


    # Encoder1 for LONGLIN
    """
    python run_table_bench_encoder_eval.py --decoder_model_path /data4/sft_output/qwen2-base-0802/checkpoint-2400 \
    --model_type "1" \
    --encoder_model_path /data0/gxj/sft_checkpoints/20col_-1/lr1e-5_constant_with_warmup_bs1024_bf16_freezedecoder_table4_nods_new/checkpoint-378 \
    --patch_model_path /data0/pretrained-models/all-MiniLM-L6-v2
    """

    # Encoder2 for LIYAO
    """
    python run_table_bench_encoder_eval.py --model_type "2" \
    --encoder_model_path /data0/workspace/liliyao/saved_models/checkpoint-364 \
    --decoder_model_path /data4/sft_output/qwen2-base-0802/checkpoint-2400 \
    --patch_model_path /data0/workspace/liliyao/saved_models/projector-0812/projector.bin \
    """

