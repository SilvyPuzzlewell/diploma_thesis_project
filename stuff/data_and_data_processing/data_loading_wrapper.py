from __future__ import absolute_import

import json
import math
import sys
from pickle import load

import pandas as pd

from stuff.global_utils.equation_converter_mine import prefix2postfix, evalPostfix, postToPre, eval_prefix
from stuff.global_utils.general import is_number
from stuff.global_utils.utils import replace_numbers, get_abs_project_path

asdiva_or_folder = "svamp_paper_csvs/cv_asdiv-a/fold"
mawps_or_folder = "svamp_paper_csvs/cv_mawps/fold"

asdiva_mine_folder = "svamp_paper_csvs/new_asdiv-a_cs/fold"
wp500cz_folder = "svamp_paper_csvs/new_WP500CZ/fold"
mawps_mine_folder = "svamp_paper_csvs/new_mawps_cs/fold"
svamp_mine_folder = "svamp_paper_csvs/new_svamp_cs/fold"

griffiths_train_all_bin_path =f"{get_abs_project_path()}/stuff/griffiths_solver_vunedited/data/train_all_postfix.p"
griffiths_train_mawps_bin_path = f"{get_abs_project_path()}/stuff/griffiths_solver_vunedited/data/train_mawps_postfix.p"
griffiths_test_mawps_bin_path = f"{get_abs_project_path()}/stuff/griffiths_solver_vunedited/data/test_mawps_postfix.p"

en_fold = "EN"

def load_dataset_from_csv(dir, individual=False):
    train = pd.read_csv(f"{dir}0/train.csv")
    dev = pd.read_csv(f"{dir}0/dev.csv")
    if not individual:
        return pd.concat([train, dev])
    else:
        return train, dev

def get_mawps_mine_cs(individual=False):
    return load_dataset_from_csv(mawps_mine_folder, individual)

def get_asdiva_mine_cs(individual=False):
    return load_dataset_from_csv(asdiva_mine_folder, individual)

def get_svamp_mine_cs(individual=False):
    return load_dataset_from_csv(svamp_mine_folder, individual)

def get_mawps_mine_en(individual=False):
    return load_dataset_from_csv(f"{mawps_mine_folder}{en_fold}", individual)

def get_asdiva_mine_en(individual=False):
    return load_dataset_from_csv(f"{asdiva_mine_folder}{en_fold}", individual)

def get_svamp_mine_en(individual=False):
    return load_dataset_from_csv(f"{svamp_mine_folder}{en_fold}", individual)

def get_wp500cz(individual=False):
    return load_dataset_from_csv(wp500cz_folder, individual)


def get_mawps_or(individual=False):
    return load_dataset_from_csv(mawps_or_folder, individual)

def get_asdiva_or(individual=False):
    return load_dataset_from_csv(asdiva_or_folder, individual)


def get_math23k_or(individual=False):
    return load_dataset_from_csv("svamp_paper_csvs/Math23k/foldEN", individual)
def get_math23k_enb(individual=False):
    return load_dataset_from_csv("svamp_paper_csvs/Math23k/fold", individual)
def get_math23k_enh(individual=False):
    return load_dataset_from_csv("svamp_paper_csvs/Math23k_helsinki/fold", individual)
def get_math23k_cz(individual=False):
    return load_dataset_from_csv("svamp_paper_csvs/Math23k_cs/fold", individual)


def get_griffiths_all_train():
    griffiths_data = load_data_from_griffiths_binary(griffiths_train_all_bin_path)
    csv = transform_from_griffits(griffiths_data)
    return csv

def get_griffiths_mawps_train():
    griffiths_data = load_data_from_griffiths_binary(griffiths_train_mawps_bin_path)
    csv = transform_from_griffits(griffiths_data)
    return csv

def get_griffiths_mawps_test():
    griffiths_data = load_data_from_griffiths_binary(griffiths_test_mawps_bin_path)
    csv = transform_from_griffits(griffiths_data)
    return csv



def load_data_from_griffiths_binary(absolute_path):
    # Get the lines in a binary as list
    with open(absolute_path, "rb") as fh:
        file_data = load(fh)

    return file_data


def transform_to_griffits(path, allow_wrong_answers=True):
    df = pd.read_csv(path)
    ret = []
    for index, row in df.iterrows():
        question = row['Question']
        numbers = row['Numbers']
        if numbers[0] == '[':
            numbers = eval(row['Numbers'])
        else:
            numbers = [float(x) for x in numbers.split()]
        equation = row['Equation']
        answer = row['Answer']
        global cur_numbers
        cur_numbers = numbers

        equation = prefix2postfix(equation.split())
        #twogroups_custom_sub(match_pattern, replace_template, sentence)
        equation = replace_numbers(equation, numbers=numbers)
        val = evalPostfix(equation)
        answer_num = float(answer)
        if not allow_wrong_answers:
            assert math.isclose(val, answer_num, rel_tol=1e-2) #sanity check

        gr_question = ('question', replace_numbers(question, numbers=numbers))
        gr_equation = ('equation', equation)
        gr_answer = ('answer', str(answer))

        cur = (gr_question, gr_equation, gr_answer)
        ret.append(cur)

    return ret

def get_and_replace_numerals(sentence):
    num_counter = 0
    nums = []
    modified_sentence = []
    if isinstance(sentence, str):
        arr = sentence.split()
    else:
        arr = sentence
    for w in arr:
        if is_number(w):
            nums.append(float(w))
            modified_sentence.append(f"number{num_counter}")
        else:
            modified_sentence.append(w)

    return " ".join(modified_sentence), " ".join([str(num) for num in nums])



def transform_from_griffits(g_data):
    """
    should return csv data in format [Question, Numbers, Equation, Answer]
    :return:
    """
    dat = []
    for d in g_data:
        cur = [None for _ in range(4)]
        for tup in d:
            if tup[0] == 'equation':
                eq = tup[1][4:]
                postfix_val = evalPostfix(eq)
                eq = postToPre(eq)
                prefix_val = eval_prefix(eq)
                assert postfix_val == prefix_val #sanity check
                eq, _ = get_and_replace_numerals(eq)
                cur[2] = eq
            elif tup[0] == 'answer':
                cur[3] = tup[1]
            elif tup[0] == 'question':
                ques, nums = get_and_replace_numerals(tup[1])
                cur[1] = nums
                cur[0] = ques
            else:
                print("something wrong!")
                sys.exit(1)
        dat.append(cur)
    columns = ["Question", "Numbers", "Equation", "Answer"]
    df = pd.DataFrame(dat, columns=columns)
    return df


def _transform_from_griffits_test():
    griffiths_train_all_bin_path = f"{get_abs_project_path()}/stuff/griffiths_solver_vunedited/data/train_all_postfix.p"
    griffiths_test_all_bin_path = f"{get_abs_project_path()}/stuff/griffiths_solver_vunedited/data/test_mawps_postfix.p"
    griffiths_data_train = load_data_from_griffiths_binary(griffiths_train_all_bin_path)
    griffiths_data_test = load_data_from_griffiths_binary(griffiths_test_all_bin_path)
    transform_from_griffits(griffiths_data_train)

#_transform_from_griffits_test()

MATH23K_ORIGINAL_FILE =f"{get_abs_project_path()}/stuff/data_and_data_processing/datasets/Math23K/Math_23K_better_ed.json"
MATH23K_TRANSLATED_FILE = f"{get_abs_project_path()}/stuff/data_and_data_processing/datasets/Math23K/baidu_translate/math23k_baidu_trns_all.json"


def get_math23k_json():
    with open(MATH23K_ORIGINAL_FILE, encoding='utf-8-sig') as fh:
        dtn = fh.read()
        dtn = dtn.replace("}", "},")
        dtn = dtn[:-3] + dtn[-2:]
        data = json.loads(dtn)
    return data

def get_translated_math23k_json():
    with open(MATH23K_TRANSLATED_FILE, encoding='utf-8-sig') as fh:
        data = json.load(fh)
    return data

