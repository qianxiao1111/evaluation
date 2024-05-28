#!/bin/bash
PRED_TABLES_PATH="preds/pred_tables.json"
LABEL_TABLES_PATH="data/y_tables.json"
PRED_COLUMNS_PATH="preds/pred_columns.json"
LABEL_COLUMNS_PATH="data/y_columns.json"

python eval.py \
    --pred_tables "$PRED_TABLES_PATH" \
    --label_tables "$LABEL_TABLES_PATH" \
    --pred_columns "$PRED_COLUMNS_PATH" \
    --label_columns "$LABEL_COLUMNS_PATH"