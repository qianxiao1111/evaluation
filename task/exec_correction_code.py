# 离线执行 code correction 的 code
# 检验结果一致性
import os
import sys

sys.path.append(".")
import glob
import random
import warnings
import pandas as pd
from task.utils.load import load_json, save_json
from evaluate_code_correction.utils import recraft_query, extract_code_without_comments

warnings.filterwarnings("ignore")


def save_script(code, path_to_save):
    with open(path_to_save, "w") as f:
        f.write(code)


def get_locals_from_path(table_paths):
    if len(table_paths) == 1:
        df = pd.read_csv(table_paths[0], low_memory=False)
    else:
        df = [pd.read_csv(path, low_memory=False) for path in table_paths]
    if isinstance(df, pd.DataFrame):
        locals = {"df": df}
    else:
        locals = {}
        for i, dataframe in enumerate(df):
            locals[f"df{i + 1}"] = dataframe
    return locals


def gen_script_path(path_to_result, i):
    par_dir, file_name = os.path.split(path_to_result)
    grad_par_dir = os.path.dirname(par_dir)
    new_par_dir = os.path.join(grad_par_dir, "code_script")
    new_file_name = file_name.replace(".json", "-{}.py".format(i))
    return os.path.join(new_par_dir, new_file_name)


def add_local_df(code, table_paths):
    code_prefix = "import pandas as pd\n"
    if len(table_paths) == 1:
        code_prefix += "df = pd.read_csv('{}')\n".format(table_paths[0])
    else:
        for i in range(len(table_paths)):
            code_prefix += "df{} = pd.read_csv('{}')\n".format(i + 1, table_paths[i])
    return code_prefix + code


summary = []
path_to_results = "/home/dev/zhangga/datasets/correction_results/reuslts/"
for path_to_result in glob.glob(path_to_results + "results_*.json"):
    samples = load_json(path_to_result)
    for i in random.sample(range(len(samples)), 5):
        sample = samples[i]
        table_paths = [
            os.path.join(os.getcwd(), path) for path in sample["table_paths"]
        ]
        observe = sample["observe"]
        code = sample["code"]
        if observe != "Code Error: output empty code.." and len(observe) < 1000:
            try:
                pure_code = extract_code_without_comments(code)
                pure_code = add_local_df(pure_code, table_paths)
                locals = get_locals_from_path(table_paths)
                pure_code = recraft_query(pure_code, locals)
                script_path = gen_script_path(path_to_result, i)
                save_script(pure_code, script_path)
                offfline_observe = ""
            except Exception as e:
                print(e)
                script_path = ""
                offfline_observe = str(e)
            summary.append(
                {
                    "path_to_result": path_to_result,
                    "script_path": script_path,
                    "observe": observe,
                    "offfline_observe": offfline_observe,
                }
            )

save_json("/home/dev/zhangga/datasets/correction_results/reuslts/summary.json", summary)
