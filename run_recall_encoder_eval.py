import os 
import re
import torch 
import pandas as pd 
from utils import load_json, get_table_infos
from encoder_models.encoder1.build_input import build_instruction, build_question 
from encoder_models.encoder2.build_input import build_encoder_recall_input
from recall_eval.run_eval import (
    parser_list,
    eval_outputs,
    save_result,
    make_pred,
    pprint_format
)
from tqdm import tqdm

def format_inputs(samples, tokenizer, model_type):
    msgs = []
    encoder_inputs = []
    for sample in samples:
        csv_paths = sample["table_paths"]
        query = sample["query"]
        df_names = sample["df_names"]
        table_info = sample["table_infos"]
        instruction = build_question(csv_paths, df_names, query)
        decoder_input = build_instruction(instruction, tokenizer)
        msgs.append(decoder_input)
        if model_type == "1":
            encoder_input = None
        else:
            encoder_input = build_encoder_recall_input(table_info)
        encoder_inputs.append(encoder_input)
    
    return msgs, encoder_input

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
    for i , (decoder_inp, encoder_inp) in tqdm(enumerate(list(zip(decoder_inputs, encoder_inputs)))):
        output = {}
        query = test_datas[i]["query"]
        table_path = test_datas[i]["table_paths"]
        df_names = test_datas[i]["df_names"]

        if model_type == "1":
            model_output = model.generate(
                decoder_inp,
                max_new_tokens=max_new_tokens, 
                eos_token_id = model.tokenizer.eos_token_id, 
                pad_token_id = model.tokenizer.eos_token_id,
                temperature = temperature,
                do_sample=True
            )
            output_content = model.tokenizer.decode(model_output[0], skip_special_tokens=True).strip()
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    if model_type =="1":
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
    pred = parser_list(model_outputs)

    report = eval_outputs(pred, samples)
    preds = make_pred(samples, pred)
    # save result
    pprint_format(report)
    save_result(preds, report, eval_results_save_path)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import shutil
    output_dir = Path(__file__).parent / "images"
    if os.path.exists(output_dir):
        if not os.access(output_dir, os.W_OK):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            os.chmod(output_dir, 0o777)
            print("not write permission, makedir:", output_dir)
        else:
            print(f"{output_dir} exists!")
    else:
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o777)
        print("makedir:", output_dir)
    
    parser = argparse.ArgumentParser(description="eval tableqa python code")

    # parser.add_argument(
    #     "--temperature", type=float, default=0.01, help="Temperature setting"
    # )

    parser.add_argument(
        "--decoder_model_path",
        type=str,
        # required=True,
        default="/data4/sft_output/qwen2-base-0717/checkpoint-2000",
        help="Decoder base model path.",
    )

    parser.add_argument(
        "--patch_model_path",
        type=str,
        default="/data0/workspace/tjm/review_code/checkpoints/cp271-qwen-ckpt_2000-only_proj_2e4_2epoch_table/projector.bin ",
        help="Patch model path, for encoder1 this path is sentence transformer path, for encoder2 this path is projector weigtht path",
    )

    parser.add_argument(
        "--encoder_model_path",
        type=str,
        default="/data0/workspace/liliyao/saved_models/checkpoint-271",
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
        default="evalset/retrieval_test/recall_set.json",
        help="Test Set Path",
    )

    parser.add_argument(
        "--eval_results_save_path",
        type=str,
        default="output/encoder_recall_1.json",
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
        "--temperature", type=float, default=0, help="Temperature setting"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of output new tokens",
    )

    args = parser.parse_args()
    main(args)

    # Encoder1 for LONGLIN
    """
    python run_tableqa_encoder_eval.py --decoder_model_path /data4/sft_output/qwen2-base-0727/ \
    --model_type "1" \
    --encoder_model_path /home/llong/gxj/code/checkpoints/sft/30col_-1_contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717/lr_1e-5_bs1024_bf16_freezedecoder_constantlr/checkpoint-340 \
    --patch_model_path /data0/pretrained-models/all-MiniLM-L6-v2
    """
    # Encoder2 for LIYAO
    """
    python run_tableqa_encoder_eval.py --decoder_model_path /data4/sft_output/qwen2-base-0727/ \
    --model_type "2" \
    --encoder_model_path /data0/workspace/liliyao/saved_models/checkpoint-271 \
    --patch_model_path /data0/workspace/tjm/review_code/checkpoints/cp271-qwen-ckpt_2000-only_proj_2e4_2epoch_table/projector.bin
    """