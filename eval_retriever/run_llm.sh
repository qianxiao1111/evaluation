#!/bin/bash
URL_GEN="http://localhost:8081"
URL_EXTRACT="http://localhost:8081"
NUM=3


python eval_llm.py \
    --gen_model_url "$URL_GEN" \
    --extract_model_url "$URL_EXTRACT" \
    --num $NUM