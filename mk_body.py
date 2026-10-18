import json
body = {
    "name": "electronic-neuron-micro-cluster",
    "description": "小模型 + 外部记忆的本地知识系统 | 知识问答 80.0% | 6 小时压测 12 万条 | 8 GB 显存单机",
    "private": False,
    "has_wiki": False,
    "has_issues": True,
}
open(r"E:/aqca/work/mem-neuron\repo_body.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
print("body written")