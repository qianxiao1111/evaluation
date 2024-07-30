import pandas as pd 
from encoder_models.encoder1.config import INSERT_SEP_TOKEN, INSERT_EMBS_TOKEN,INSERT_SEP_TOKEN

def dataframe_info_simple(df: pd.DataFrame, df_name:str, comments=None):
    """
    根据 dataframe 获取 dataframe description 信息
    :param df: 输入 dataframe
    :param df_name: dataframe name
    :param comments: 列名的备注信息, dict
    :return: 能反馈 dataframe 的信息
    """
    # df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe include the data types, comments, the column values info as follows:\n{desc_info}\n*/"""
    df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe that can be used as follows:\n{desc_info}\n*/"""
    # df_info_template_simple = """/*\n'{df_name}' each column information:\n {desc_info}\n*/"""
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    # 添加 Example Values 列，使用所有唯一值但最多取三个
    if comments is not None:
        # 将comments转换为一个字典，以便快速查找
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        # 为每一列添加comment和info信息
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

        # info_df['Comment'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        # info_df['Info'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("info", ""))
    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    desc_info_lines = []
    for key, value in desc_info_dict.items():
        comment = value.get("Comment", "")
        if comment:
            comment_str = "means " + comment + "."
        else:
            comment_str = ""

        data_type = value["Data Type"]
        
        contains_nan = value["Contains NaN"]
        if contains_nan:
            contains_nan_str = "contains NaN, "
        else:
            contains_nan_str = ""
        
        is_unique = value["Is Unique"]
        if is_unique:
            unique_str = "is unique, "
        else:
            unique_str = ""
            # unique_str = "is not unique, "
        

        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        dil = f"- '{key}' {data_type}, {unique_str}{contains_nan_str}{comment_str} Example Values: {INSERT_SEP_TOKEN + INSERT_EMBS_TOKEN + INSERT_SEP_TOKEN}"
        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info


def build_instruction(prompt, tokenizer):
    """
    Apply the chat template to the user prompt

    Args:
        prompt (str): The user prompt.
        tokenizer: The tokenizer object.

    Returns:
        str: The instruction text.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    decoder_input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return decoder_input_text

def build_question(csv_paths,df_names, query):
    """
    Build the instruction text for the user question.

    Args:
        conv (dict): A dictionary containing conversation information. It should contain the following keys: csv_abs_paths, df_names, query.

    Returns:
        str: The generated question string.

    """
    
    pref = '''With several pandas dataframes available, your task is to write the Python code to address the user's question.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Evaluate the dataframes and the question to determine the solution.\nPython code: Generate the Python code, within ```python ... ```.\n\n## Details about the dataframes:\n\n'''    
    # csv_paths, df_names = conv['csv_abs_paths'], conv['df_names']
    df_list = [pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False,
        nrows=500
    ) for path in csv_paths]
    df_info_list = [dataframe_info_simple(df, df_name) for df, df_name in zip(df_list, df_names)]
    suf = '''\n\nQuestion: ''' + query + '\n'
    return pref + '\n\n'.join(df_info_list) + suf