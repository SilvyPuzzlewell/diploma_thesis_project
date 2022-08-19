import json
from sacremoses import MosesTokenizer, MosesDetokenizer

with open("ASDiv-a.json") as in_, open("ASDiv-a.txt", "w", encoding='utf-8') as out:
    data = json.load(in_)
    for problem in data["ProblemSet"]:
        print(problem["Body"]+"\n"+problem["Question"]+"\n"+problem["Answer"],file=out)

md = MosesDetokenizer()
with open("mawps.json") as in_, open("cs_mawps.txt", "w", encoding='utf-8') as out:
    data = json.load(in_)
    for problem in data:
        print(md.detokenize(problem["sQuestion"].split(" ")),file=out)

md = MosesDetokenizer()
with open("SVAMP.json") as in_, open("SVAMP.txt", "w", encoding='utf-8') as out:
    data = json.load(in_)
    for problem in data:
        print(problem["Body"]+"\n"+problem["Question"],file=out)
