import json

with open("cs_ASDiv-a.txt", encoding="utf-8") as in_, open("ASDiv-a.json") as in_json, open("cs_ASDiv-a.json", 'w',
                                                                                            encoding="utf-8") as out:
    asdiva = json.load(in_json)
    i = 0
    for line in in_:
        line = line.strip()
        if i % 3 == 0:
            asdiva["ProblemSet"][i // 3]["BodyEN"] = asdiva["ProblemSet"][i // 3]["Body"]
            asdiva["ProblemSet"][i // 3]["Body"] = line
        if i % 3 == 1:
            asdiva["ProblemSet"][i // 3]["QuestionEN"] = asdiva["ProblemSet"][i // 3]["Question"]
            asdiva["ProblemSet"][i // 3]["Question"] = line
        if i % 3 == 2:
            asdiva["ProblemSet"][i // 3]["AnswerEN"] = asdiva["ProblemSet"][i // 3]["Answer"]
            asdiva["ProblemSet"][i // 3]["Answer"] = line
        i += 1
    json.dump(asdiva, out, ensure_ascii=False)

with open("cs_svamp.txt", encoding="utf-8") as in_, open("SVAMP.json") as in_json, open("cs_svamp.json", 'w',
                                                                                        encoding="utf-8") as out:
    asdiva = json.load(in_json)
    i = 0
    for line in in_:
        line = line.strip()
        if i % 2 == 0:
            asdiva[i // 2]["BodyEN"] = asdiva[i // 2]["Body"]
            asdiva[i // 2]["Body"] = line
        if i % 2 == 1:
            asdiva[i // 2]["QuestionEN"] = asdiva[i // 2]["Question"]
            asdiva[i // 2]["Question"] = line
        i += 1
    json.dump(asdiva, out, ensure_ascii=False)

with open("cs_mawps.txt", encoding="utf-8") as in_, open("mawps.json") as in_json, open("cs_mawps.json", 'w',
                                                                                        encoding="utf-8") as out:
    asdiva = json.load(in_json)
    i = 0
    for line in in_:
        line = line.strip()
        asdiva[i]["sQuestionEN"] = asdiva[i]["sQuestion"]
        asdiva[i]["sQuestion"] = line
        i += 1
    json.dump(asdiva, out, ensure_ascii=False)
