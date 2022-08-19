import json


def is_number(word):
    try:
        float(word)
    except:
        return False
    return True

def get_json_from_processed_data(dataset):
    json_list = []
    for dat in dataset:
        dict = {}
        dict["question"] = dat[0][1]
        dict["equation"] = dat[1][1]
        dict["answer"] = dat[2][1]
        json_list.append(dict)
    return json_list

def dump_dataset(dataset_json, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=4)

#datasets can't contain same problems
def check_integrity(train_dataset, test_dataset):
    dt1 = get_json_from_processed_data(train_dataset)
    dt2 = get_json_from_processed_data(test_dataset)
    counter = 0
    for tr_problem in dt1:
        for tst_problem in dt2:
            if tr_problem == tst_problem:
                return [tr_problem, tst_problem]
    return None

