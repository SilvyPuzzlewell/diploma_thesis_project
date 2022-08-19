import json
import os
import pickle
import sys

from Naked.toolshed.shell import execute_js, muterun_js

os.system("node baidu_js_api/test.js")
"""
response = muterun_js('baidu_js_api/test.js')
if response.exitcode == 0:
  print(response.stdout)
else:
  sys.stderr.write(response.stderr)


with open("../Math23K/Math23K_read.json") as fh:
    json_data = json.load(fh)

from googletrans import Translator
translator = Translator()

counter = 2
for problem in json_data:
    txt = problem['segmented_text']
    translate = translator.translate(txt)
    txt_trs = translate.text
    problem['en_text'] = txt_trs
    counter += 1
    if counter == 100:
        break

with open('../Math23K/math23k_trns.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
"""