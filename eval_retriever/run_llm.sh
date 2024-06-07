#!/bin/bash
URL_GEN="http://localhost:8083"
URL_EXTRACT="http://localhost:8083"
NUM=3


python eval_retriever/eval_llm.py \
    --gen_model_url "$URL_GEN" \
    --extract_model_url "$URL_EXTRACT" \
    --num $NUM