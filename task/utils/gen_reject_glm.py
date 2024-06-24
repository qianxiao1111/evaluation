import sys

sys.path.append(".")
import warnings
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from task.gen import RejectChain
from task.utils.load import load_json, save_json


warnings.filterwarnings(action="ignore")
llm = ChatOpenAI(
    temperature=0.01,
    model="glm-4",
    openai_api_key="396366d4726ab616ad7233b5cb2f4ba6.x0dtGn7E9YNGkudv",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)
filepath = "evalset/reject_test/ground_truth.json"
# start
rejector = RejectChain.from_llm(llm)
samples = load_json(filepath)
result = []
for sample in tqdm(samples):
    is_reject = None
    resp = rejector.predict(df_info=sample["df_info"], query=sample["query"])
    if "yes" in resp.lower():
        is_reject = False
    elif "no" in resp.lower():
        is_reject = True
    else:
        pass
    result.append(
        {
            "df_info": sample["df_info"],
            "query": sample["query"],
            "is_reject": sample["is_reject"],
            "is_reject_glm": is_reject,
        }
    )
    save_json("datasets/20240624/ground_truth_with_glm.json", result)
