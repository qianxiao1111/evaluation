import os 
import pandas as pd
import datetime
import torch

from utils import load_json, save_json
from encoder_models.encoder1.build_input import build_instruction, dataframe_info_simple
from encoder_models.encoder1.format import dataframe_info_combined
from encoder_models.encoder2.build_input import build_encoder_input
from tqdm import tqdm

from evaluate_code_correction.prompt import (
    RECTIFY_PROMPT_PYTHON_SYSTEM,
    RECTIFY_PROMPT_PYTHON_INSTRUCTION
)
from evaluate_code_correction.run_eval import eval_outputs, run_eval

def build_code_correction_question(csv_paths,df_names, query, ori_code, observe, cot):
    template = "\n".join([RECTIFY_PROMPT_PYTHON_SYSTEM, RECTIFY_PROMPT_PYTHON_INSTRUCTION])
    output = cot + "Python Code:\n" + ori_code
    current_time = datetime.datetime.now().strftime("%Y-%m-%d:%H")
    df_list = [pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False,
        nrows=500
    ) for path in csv_paths]
    df_info_list = [dataframe_info_combined(df, df_name) for df, df_name in zip(df_list, df_names)]
    table_infos = "\n\n".join(df_info_list)
    template = template.format(
        query=query,
        output=output,
        observe=observe,
        table_infos=table_infos,
        current_time=current_time
    )
    return template


def format_inputs(samples, tokenizer, model_type):
    msgs = []
    encoder_inputs = []
    for sample in samples:
        csv_paths = sample["table_paths"]
        query = sample["query"]
        df_names = [f"df{i+1}" for i in range(len(csv_paths))]
        table_info = sample["table_infos"]
        ori_code = sample["code"]
        cot = sample["cot"]
        observe = sample["observation"]
        instruction = build_code_correction_question(csv_paths, df_names, query, ori_code, observe, cot)
        decoder_input = build_instruction(instruction, tokenizer)
        msgs.append(decoder_input)
        if model_type == "1":
            encoder_input = None
        else:
            encoder_input = build_encoder_input(table_info)
        encoder_inputs.append(encoder_input)
    
    return msgs, encoder_inputs

@torch.inference_mode()
def generate_outputs(
    model, 
    test_datas,
    decoder_inputs, 
    encoder_inputs,
    temperature,
    max_new_tokens,
    device: str = "cuda",
    model_type: str = "1",
):
    model.eval().to(device)
    model_outputs = []
    for i, (decoder_inp, encoder_inp) in tqdm(enumerate(zip(decoder_inputs, encoder_inputs)), total=len(decoder_inputs)):
        output = {}
        query = test_datas[i]["query"]
        table_path = test_datas[i]["table_paths"]
        df_names = [f"df{i+1}" for i in range(len(table_path))]

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
            # output_content = model.tokenize0r.decode(model_output[0], skip_special_tokens=True).strip()
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
        output["output_text"] = output_content
        output["query"] = query
        output["df_names"] = df_names
        output["table_paths"] = table_path
        output["instruction"] = decoder_inp
        output["encoder_input"] = encoder_inp
        model_outputs.append(output)
    return model_outputs


def main(args):
    """main function"""
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
    if args.run_llm_eval:
        from evaluate_code_correction.llms import llm_judge
        llm_for_judge = llm_judge
    else:
        llm_for_judge = None

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
    
    samples = load_json(eval_dataset_path)

    if num_samples is not None:
        samples = samples[: num_samples]
    
    decoder_inputs, encoder_inputs = format_inputs(samples, tokenizer, model_type)
    print("Generating eval answers now..")
    model_outputs = generate_outputs(
        test_datas=samples,
        model=model,
        decoder_inputs=decoder_inputs,
        encoder_inputs=encoder_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
        model_type=model_type
    )
    print("Generating answers finished..")
    eval_answers = eval_outputs(
        model_outputs,
        eval_dataset_path,
        lan_type="Python",
        test_csv_file_path="./",
        num_samples=num_samples
    )
    print("Eval answers construct complete..")
    save_json(eval_results_save_path, eval_answers)
    run_eval(eval_result_path=eval_results_save_path, llm_for_judge=llm_for_judge)


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
        default="evalset/code_correction_test/correction_set.json",
        help="Test Set Path",
    )

    parser.add_argument(
        "--eval_results_save_path",
        type=str,
        default="output/encoder_code_correction_2.json",
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
        default=None,
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
    python run_code_correction_encoder_eval.py --decoder_model_path /data4/sft_output/qwen2-base-0802/checkpoint-2400 \
    --model_type "1" \
    --encoder_model_path /data0/gxj/sft_checkpoints/20col_-1/lr1e-5_constant_with_warmup_bs1024_bf16_freezedecoder_table4_nods_new/checkpoint-378 \
    --patch_model_path /data0/pretrained-models/all-MiniLM-L6-v2
    """

    # Encoder2 for LIYAO
    """
    python run_code_correction_encoder_eval.py --model_type "2" \
    --encoder_model_path /data0/workspace/liliyao/saved_models/checkpoint-364 \
    --decoder_model_path /data4/sft_output/qwen2-base-0802/checkpoint-2400 \
    --patch_model_path /data0/workspace/liliyao/saved_models/projector-0812/projector.bin \
    """