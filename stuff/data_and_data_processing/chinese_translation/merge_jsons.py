import json

jsons = []
filename = "math23k_baidu_trns_all"

with open(f"../datasets/Math23K/baidu_translate/old/math23k_trns.json", 'r') as fh:
    jsons += json.load(fh)

for i in range(7,23):
    with open(f"../Math23K/baidu_translate/old/math23k_trns_{i*1000}_{(i+1)*1000}.json", 'r') as fh:
        json_data = json.load(fh)
        jsons += json_data

with open(f"../datasets/Math23K/baidu_translate/old/math23k_trns_23000_23162.json", 'r') as fh:
    json_data = json.load(fh)
    jsons += json_data

with open(f"../Math23K/baidu_translate/{filename}.json", 'w', encoding='utf-8') as f:
    json.dump(jsons, f, ensure_ascii=False, indent=4)
