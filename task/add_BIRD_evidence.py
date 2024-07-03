import sys

sys.path.append(".")
from task.utils.load import load_json, save_json

path_to_bird = "/home/dev/zhangga/datasets/BIRD_dev/dev.json"
bird = load_json(path_to_bird)
# code
# path_to_samples = "evalset/code_correction_test/correction_set.json"
# samples = load_json(path_to_samples)
# new = []
# cnt = 0
# for sample in samples:
#     if "BIRD_dev" in sample["table_paths"][0]:
#         temp = [temp for temp in bird if temp["question"] == sample["query"]][0]
#         if temp["evidence"] != "":
#             sample["query"] = sample["query"] + "\n{}\n".format(temp["evidence"])
#             cnt += 1
#     new.append(sample)
# print(cnt)
# print(cnt / len(samples))
# save_json("evalset/code_correction_test/correction_set_evidence.json", new)


# recall
path_to_samples = "evalset/retrieval_test/recall_set.json"
samples = load_json(path_to_samples)
new = []
cnt = 0
for sample in samples:
    temp = [temp for temp in bird if temp["question"] == sample["query"]]
    if len(temp) > 0:
        temp = temp[0]
        if temp["evidence"] != "":
            sample["query"] = sample["query"] + "\n{}\n".format(temp["evidence"])
            cnt += 1
    new.append(sample)
print(cnt)
print(cnt / len(samples))
save_json("evalset/retrieval_test/recall_set_evidence.json", new)
