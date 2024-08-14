

def build_encoder_input(table_info: str):

    instruction = "1. Extract the table-level embedding of the input table.\n2. Only generate SQL to answer the following question relevant to the input table, and do not generate anything else."

    encoder_input_text_template = f"### Instruction:\n{instruction}\n\n### Input:\n"
    encoder_input_text = encoder_input_text_template + table_info + '\n\n'

    return encoder_input_text

def build_encoder_recall_input(table_info: str):
    instruction = "1. Extract the table-level embedding of the input table.\n2. Your task is to analyze the given tables and columns, and choose most relevant tables and columns to the user's query.Answer in Format: tables is: ['table_name1','table_name2',...];columns is: ['table_name1.column_name1','table_name2.column_name2',...]"

    encoder_input_text_template = f"### Instruction:\n{instruction}\n\n### Input:\n"
    encoder_input_text = encoder_input_text_template + table_info + '\n\n'

    return encoder_input_text

def build_encoder_reject_input(table_info):
    instruction = "1. Extract the table-level embedding of the input table.\n2. Only generate `yes` or `no` to answer whether the following question is relevant to the input table, and do not generate anything else."

    encoder_input_text_template = f"### Instruction:\n{instruction}\n\n### Input:\n"
    encoder_input_text = encoder_input_text_template + table_info + '\n\n'

    return encoder_input_text
