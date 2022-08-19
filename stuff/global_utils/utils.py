from __future__ import absolute_import

import copy
import json
import os
import re

from re import sub
from pickle import dump

# Data constraints
from word2number import w2n
import pickle
MAX_LENGTH = 60

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


def save_file(data, file):
    pickle.dump(data, open(file, 'wb'))

def load_file(file):
    return pickle.load(open(file, 'rb'))

def en_number_tag(text):
    sent = []
    for word in text.split(' '):
        if word.lower() == 'twice':
            sent.append(" 2 times ")
        else:
            try:
                sent.append(str(w2n.word_to_num(word)))
            except:
                sent.append(word)

    return ' '.join(sent)

cur_numbers = []
def replace_number_fun(mtch):
    global cur_numbers
    cur_number_idx = int(mtch.group(1))
    num = cur_numbers[cur_number_idx]
    return str(num)

def replace_numbers(expr, pattern="number(\d+)", numbers=None):
    if numbers:
        global cur_numbers
        cur_numbers = numbers
    expr = re.sub(pattern, replace_number_fun, expr)
    return expr


#path = '/home/sylvi/diplomka/my_code/data_and_data_processing/svamp_paper_csvs/new_ASDIVA_p_MAWPS_p_SVAMP/fold0/train.csv'
#transform_to_griffits(path)



def to_binary(absolute_path, what):
    # Save to a binary
    with open(absolute_path, 'wb') as fh:
        dump(what, fh)


def get_as_tuple(example):
    # Separate the trainable data_and_data_processing
    ex_as_dict = dict(example)

    return ex_as_dict["question"], ex_as_dict["equation"]

def get_as_answ_tuple(example):
    # Separate the trainable data_and_data_processing
    ex_as_dict = dict(example)

    return ex_as_dict["question"], str(ex_as_dict["answer"])


def expressionize(what):
    # It may help training if the 'x =' is not learned
    what = sub(r"([a-z] \=|\= [a-z])", "", what)
    return sub(r"^\s+", "", what)


def print_epoch(what, clear=False):
    # Overwrite the line to see live updated results
    print(f"{what}", end="\r")

    if clear:
        # Clear the line being overwritten by print_epoch
        print()



def _separate_str_fun(mtch):
    return f" {mtch.group(0)} "

def _replace_delimiter_fun(mtch):
    return f"{mtch.group(1)}.{mtch.group(2)}"

def _separate_2groups_fun(mtch):
    return f"{mtch.group(1)} {mtch.group(2)}"


def cz2en_delimiter(str):
    return re.sub("(\d+),(\d+)", _replace_delimiter_fun, str)

def separate_pattern(str, pattern):
    return re.sub(pattern, _separate_str_fun, str)

def separate_2groups_pattern(pattern, str):
    return re.sub(pattern, _separate_2groups_fun, str)





twogroups_custom_template = ''
def _2groups_custom_fun(mtch):
    global twogroups_custom_template
    group1 = '~1'
    group2 = '~2'
    ret = copy.deepcopy(twogroups_custom_template)
    try:
        ret = ret.replace(group1, mtch.group(1))
    except IndexError:
        pass
    try:
        ret = ret.replace(group2, mtch.group(2))
    except IndexError:
        pass
    return ret
def twogroups_custom_sub(match_pattern, replace_template, sentence):
    global twogroups_custom_template
    twogroups_custom_template = replace_template
    sentence = re.sub(match_pattern, _2groups_custom_fun, sentence)
    twogroups_custom_template = ''
    return sentence


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_abs_project_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
