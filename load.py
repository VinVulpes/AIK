import json
res = {}
with open("words","r",encoding='utf-8') as f:
    res = json.load(f)
# print(list(res)[:100])
# for i in 