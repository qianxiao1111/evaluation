import json

with open("evalset/code_correction_test/correction_set.json", "r") as f:
    samples = json.load(f)
sample = samples[44]
# print(sample["true_result"])


from evaluate_code_correction.run_eval import text_to_array

text = "[(['496_6', 'TR496_7', 'TR496_27', 'TR496_7', 'TR496_56', 'TR496_7', 'TR496_7', 'TR496_8', 'TR496_33', 'TR496_8', 'TR496_57', 'TR496_8', 'TR496_8', 'TR496_9', 'TR496_10', 'TR496_9', 'TR496_58', 'TR496_9', 'TR496_59', 'TR496_9']), ('TR499', ['TR499_1', 'TR499_2', 'TR499_2', 'TR499_3', 'TR499_2', 'TR499_4']), ('TR501', ['TR501_10', 'TR501_14', 'TR501_10', 'TR501_22', 'TR501_11', 'TR501_14', 'TR501_11', 'TR501_23', 'TR501_12', 'TR501_15', 'TR501_12', 'TR501_24', 'TR501_13', 'TR501_15', 'TR501_13', 'TR501_25', 'TR501_14', 'TR501_16', 'TR501_15', 'TR501_17', 'TR501_1', 'TR501_2', 'TR501_1', 'TR501_3', 'TR501_1', 'TR501_4', 'TR501_1', 'TR501_5', 'TR501_4', 'TR501_6', 'TR501_4', 'TR501_7', 'TR501_5', 'TR501_8', 'TR501_5', 'TR501_9', 'TR501_10', 'TR501_6', 'TR501_18', 'TR501_6', 'TR501_11', 'TR501_7', 'TR501_19', 'TR501_7', 'TR501_12', 'TR501_8', 'TR501_20', 'TR501_8', 'TR501_13', 'TR501_9', 'TR501_21', 'TR501_9']), ('TR502', ['TR502_1', 'TR502_3', 'TR502_1', 'TR502_6', 'TR502_1', 'TR502_7', 'TR502_1', 'TR502_8', 'TR502_2', 'TR502_3', 'TR502_3', 'TR502_4', 'TR502_3', 'TR502_5', 'TR502_6', 'TR502_9', 'TR502_10', 'TR502_7'])]"
a = text_to_array(text)

print(a)
