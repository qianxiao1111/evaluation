

def build_encoder_input(table_info: str):

    instruction = "1. Extract the table-level embedding of the input table.\n2. Only generate SQL to answer the following question relevant to the input table, and do not generate anything else."

    encoder_input_text_template = f"### Instruction:\n{instruction}\n\n### Input:\n"
    encoder_input_text = encoder_input_text_template + table_info + '\n\n'

    return encoder_input_text
