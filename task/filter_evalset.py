# 2024年06月24日
# 按照要求从旧版数据集中筛选数据
import os
import re
import sys
import warnings

sys.path.append(".")
warnings.filterwarnings(action="ignore")
from task.utils.load import load_json, save_json, load_df


def filter_set_recall(filepath, savepath="20240624"):
    """
    1. recall
    筛选 单表200条 多表 100条
    这里不做 random, 直接选前面的
    """
    samples = load_json(filepath)
    samples_single = [sample for sample in samples if len(sample["label_table"]) == 1]
    samples_multi = [sample for sample in samples if len(sample["label_table"]) > 1]
    samples_new = []
    samples_new.extend(samples_single[:200])
    samples_new.extend(samples_multi[:100])
    save_dir = os.path.join(savepath, "evalset/retrieval_test")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_json(os.path.join(save_dir, "recall_set.json"), samples_new)
    print("recall_set saved:{}".format(len(samples_new)))


def filter_set_reject(filepath, savepath="20240624"):
    """
    2. reject
    使用 强LLM 做投票，现有的label为 gpt4 标注
    参考 claud2、gemini 等的结果辅助判断
    update: 只有glm4 账号，先调用 utils/gen_reject_glm.p也生成结果
    """

    def tran2truth(sample):
        return {
            "df_info": sample["df_info"],
            "query": sample["query"],
            "is_reject": sample["is_reject"],
        }

    def tran2test(sample):
        return {
            "df_info": sample["df_info"],
            "query": sample["query"],
            "is_reject": "true or false",
        }

    samples = load_json(filepath)
    samples_new = [
        sample for sample in samples if sample["is_reject"] == sample["is_reject_glm"]
    ]
    samples_new = samples_new[:300]
    ground_truth = [tran2truth(sample) for sample in samples_new]
    test_query = [tran2test(sample) for sample in samples_new]
    save_dir = os.path.join(savepath, "evalset/reject_test")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_json(os.path.join(save_dir, "ground_truth.json"), ground_truth)
    print("ground_truth saved:{}".format(len(ground_truth)))
    save_json(os.path.join(save_dir, "test_query.json"), test_query)
    print("test_query saved:{}".format(len(test_query)))


def filter_set_correction(filepath, savepath="20240624"):
    """
    3. correction
    """
    max_size = 5 * 1024 * 1024  # 最大文件 5M
    samples = load_json(filepath)
    correction_set = []
    for sample in samples:
        if "csv_lower" in sample["table_paths"][0]:  # 筛选 SPIDER 和 BIRD 数据集的数据
            # # 最好有 cot
            # flag_cot = True
            # if sample["cot"] == "":
            #     flag_cot = False

            # 检查文件存在与大小
            flag_size = True
            for path in sample["table_paths"]:  # 检查 csv 的文件大小, 要求小于5M
                if not os.path.exists(path):
                    flag_size = False  # 找不到文件，无法继续

                else:
                    size = os.path.getsize(path)
                    if size > max_size:
                        flag_size = False  # 出现大文件，无法继续

            # 检查代码
            code = sample["code"]
            flag_code = True
            # 如果出现 matplotlib、sklearn、scipy 等包的调用就 pass
            if ("matplotlib" in code) | ("sklearn" in code) | ("scipy" in code):
                flag_code = False

            if flag_size & flag_code:
                # 删除不必要的代码
                # 如 read_csv \ pd.DataFrame 等形式
                pattern1 = r"\b\w+\s*=\s*pd\.read_\w+\(.*?\)"  # 正则表达式匹配形如 `任何变量名 = pd.read_任何函数名(任何字符串)` 的模式
                pattern2 = r"\s*\w+\s*=\s*pd\.DataFrame\s*\([^)]*\)"  #
                code = re.sub(pattern1, "", code)
                code = re.sub(pattern2, "", code)

                # 修改 table_paths  重新保存
                new_paths = []
                for path in sample["table_paths"]:
                    df = load_df(path)
                    new_path = path.replace("csv_lower/", "")  # json 仅删除 csv_lower
                    if not os.path.exists(
                        os.path.join(savepath, os.path.dirname(new_path))
                    ):  # 实际存储的时候要加上上级目录 save_path
                        os.mkdir(os.path.join(savepath, os.path.dirname(new_path)))
                    df.to_csv(os.path.join(savepath, new_path), index=False)
                    new_paths.append(new_path)
                sample["code"] = code
                sample["table_paths"] = new_paths
                correction_set.append(sample)
            else:
                pass
        else:
            pass
    save_dir = os.path.join(savepath, "evalset/code_correction_test")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_json(os.path.join(save_dir, "correction_set.json"), correction_set)
    print("correction_set saved:{}".format(len(correction_set)))


if __name__ == "__main__":
    savepath = "datasets/20240624"
    path_to_recall = "datasets/20240624/recall_set.json"
    filter_set_recall(path_to_recall, savepath)
    path_to_reject = "datasets/20240624/ground_truth_with_glm.json"
    filter_set_reject(path_to_reject, savepath)
    path_to_correction = "datasets/20240624/correction_set_1307.json"
    filter_set_correction(path_to_correction, savepath)
