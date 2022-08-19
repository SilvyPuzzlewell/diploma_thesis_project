from __future__ import absolute_import

import math
import os
import json
import random
import sys

import numpy as np
import pandas as pd
import re
from word2number import w2n
import sympy as sym
from sklearn.model_selection import KFold

from sympy import Eq, Pow

from stuff.data_and_data_processing.data_loading_wrapper import get_math23k_json
from stuff.data_and_data_processing.word2number_cs.number_conversion_script import word2number_cs
from stuff.global_utils.equation_converter_mine import eval_prefix, infixToPrefix
from stuff.global_utils.general import is_number
from stuff.global_utils.global_consts import NUMBER_PATTERN, OPERATORS, NUMBER_PATTERN_MINUS
from stuff.global_utils.transfer_num_copied import transfer_num_MATH23K
from stuff.global_utils.utils import replace_numbers, en_number_tag, _separate_str_fun, separate_pattern, cz2en_delimiter, \
    twogroups_custom_sub, save_file, get_abs_project_path

data_folder = "/stuff/data_and_data_processing/svamp_paper_csvs"

def one_sentence_clean(text, replace_percent=True):
    # Clean up the data_and_data_processing and separate everything by spaces
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)(?<![0-9])(\s+)?\.(\s+)?", " . ",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"(\s+)?\?(\s+)?", " ? ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"^\s+", "", text)
    text = text.replace('\n', ' ')
    text = text.replace("'", " '")
    if replace_percent:
        text = text.replace('%', ' percent')
    text = text.replace('$', ' $ ')
    text = re.sub(r"\.\s+", " . ", text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r"(?<=\d)\.$", " .", text) #add whitespace between numbers and end of sentence commas
    text = twogroups_custom_sub("(\d+)th\s", "~1 th ", text)
    sent = []
    for word in text.split(' '):
        try:
            sent.append(str(w2n.word_to_num(word)))
        except:
            sent.append(word)

    return ' '.join(sent)


def to_lower_case(text):
    # Convert strings to lowercase
    try:
        return text.lower()
    except:
        return text

def _asdiv_a_weird_nums_converter():
    pass

def _separate_dolarovky_fun(mtch):
    dol = mtch.group(2)
    num = mtch.group(1)
    if num in dolarovky_slovnik:
        num = dolarovky_slovnik[num]
    elif is_number(num):
        num = num
    else:
        sys.exit(1)
    return f"{num} {dol}"

dolarovky_slovnik = {
    "pěti": 5,
    "deseti": 10,
    "padesáti": 50
}

def _dolarovky_converter(seq):
    seq = re.sub("(?<=\s)([^\s]+)(dolarových)", _separate_dolarovky_fun, seq)
    return seq



def ASDivACZ_problem_list(path):
    print("\nWorking on cs ASDiv-a data_and_data_processing...")
    #upraveny: 513, 514, 540, 1825
    # allowed_types = {"Addition", "Subtraction", "Sum", "Common-Division", "Ceil-Division", "Floor-Division"
    #    , "Multiplication", "Difference", "TVQ-Final", "TVQ-Change", "TVQ-Initial"}
    allowed_types = {"Addition", "Subtraction", "Sum", "Common-Division",
                     "Multiplication", "Difference", "TVQ-Final", "TVQ-Change", "TVQ-Initial"}
    #forbidden_problems = {39, 152, 170, 172, 176, 178, 186, 194, 422, 432, 489, 532, 533, 536, 547, 553, 554, 555,
    #                      573, 616, 624, 636, 1113, 1114, 1115, 1116, 1118, 1125, 1128, 1352, 1356, 1383, 1505, 1506}
    forbidden_problems = {
        152, 168, 170, 172, 176, 178, 186, 194, 532, 533, 535, 536, 553, 554, 555, 573, 616, 624, 636, 1505, 1506,      #background knowledge
        422, 432, 489, 547, 1352, 1383,   #implicit deductions
        913, #nontranslated numbers
        1113, 1114, 1115, 1116, 1118, 1125, 1128,     #nontranslateble formula
        1355, 1356, #nontranslated words
        1556 #mistake in formula, is integer division logically
    }

    interesting_problems = {
        609, 610, 1741
    }  # 610 it's the type where the solver should be able to deduce from common sense that the numbers 4 205 don't mean two separate quantities
    # it shows another limitation

    # 1741 - bad translation
    problem_list = []
    problem_list_EN = []
    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)
    valid_counter = 0
    for i in range(len(json_data)):
        problem_n = i + 1
        data = json_data[i]
        # do not include nonaritmetic examples
        forbidden = False
        """
        if isinstance(data_and_data_processing["Solution-Type"], dict):
            for val in data_and_data_processing["Solution-Type"].values():
                if val not in allowed_types:
                    forbidden = True
                    break
        """
        if isinstance(data["Solution-Type"], dict):
            forbidden = True  # todo - for now, later i could use the composite problems
            # todo but apparently svamp paper isn't using them so i would make the comparisons less valid
        elif data["Solution-Type"] not in allowed_types:
            forbidden = True

        if problem_n in forbidden_problems:
            forbidden = True
        if forbidden:
            continue

        if "Body" in data and "Question" in data and "Formula" in data and "Answer" in data:
            valid_counter += 1
            def edit_equation(equation, nums):
                equation = to_lower_case(re.sub("=[0-9]+", " = x", equation))
                equation = remove_equals(equation)
                equation = separate_pattern(equation, "-")
                equation = replace_nums_equation(equation, nums)
                equation = " ".join(infixToPrefix(equation.split()))
                return equation

            body = data["Body"]
            question = data["Question"]
            problem_str = body + " " + question

            problem_str = _dolarovky_converter(problem_str)
            problem_str = separate_pattern(problem_str, "(?<=\d):(?=\d)")#separate ratios
            problem_str = _merge_thousands(problem_str)
            problem_str = word2number_cs(to_lower_case(one_sentence_clean(cz2en_delimiter(problem_str))))


            problem_str, nums, idxs = replace_nums(problem_str)
            # todo in number converter - add replace for "nulu" - 0
            # třicet, devet /nefunguje na zacatku vet? na začátku věty selhalo taky, dvaadvacet jednadvacet taky
            # petinasobek, desetinasobkem nezna
            # čtyř, dvou, deseti, sedmi, tri
            # jedenáctých
            # čtrnácti
            equation = data["Formula"]

            # equation = "10-(3+2)=5"


            # todo solution type ceil division etc. v svamp paperu se bere numericke hodnoceni bez tech r10 etc picovinek

            equation = edit_equation(equation, nums)

            answer = data["AnswerEN"]
            res = re.search(NUMBER_PATTERN, answer)
            answer = float(res.group(0))

            eval_eq(equation, nums, answer)

            problem = (problem_str, nums, equation, answer)
            problem_list.append(problem)
            if i == 139:
                continue
            equation_en = data["Formula"]
            problem_str_en = data["BodyEN"] + " " + data["QuestionEN"]
            problem_str_en = twogroups_custom_sub("(\d+)-", "~1 -", problem_str_en)
            problem_str_en = twogroups_custom_sub("(\d+),(\d+)", "~1~2", problem_str_en)
            problem_str_en = re.sub("twice(?=\s)", " 2 times ", problem_str_en, re.IGNORECASE)
            problem_str_en = to_lower_case(one_sentence_clean(problem_str_en))


            problem_str_en, nums_en, idxs_en = replace_nums(problem_str_en)
            equation_en = edit_equation(equation_en, nums_en)
            try:
                eval_eq(equation_en, nums_en, answer)
                check_en(nums_en, separate_pattern(remove_equals(data["Formula"]), "-"))
            except:
                continue
            problem_en = (problem_str_en, nums_en, equation_en, answer)
            problem_list_EN.append(problem_en)

    plen = len(problem_list)
    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")
    problem_list_filtered, cz_dupls = filter_duplicates(problem_list, ret_duplicates=True)
    problem_list_EN, en_dupls = filter_duplicates(problem_list_EN, ret_duplicates=True)
    print(f"-> Filtered {plen - len(problem_list_filtered)} duplicates, f"
          f"inally retrieved {len(problem_list_filtered)} ASDiv-a problems.")
    print(f"retrieved {len(problem_list_EN)} EN problems")
    print(f"asdiva has {valid_counter} arithmetic problems")
    return problem_list_filtered, problem_list_EN


def WP500CZ_problem_list(path):
    print("\nWorking on WP500CZ data_and_data_processing...")

    problem_list = []
    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)
    #wrong solutions in original data - problems 198, - 42 instead of 40
    #                                            231  - N2 + N2 - N3
    #                                            372  - 20 instead of 30
    #                                            443  - 125 instead of 122
    #                                            480, 481  - viz 231
    for i in range(len(json_data)):
        data = json_data[i]
        if "Problem" in data and "Equation" in data and "Solution" in data:
            problem_str = data["Problem"]
            equation = data["Equation"]
            answer = data["Solution"]

            problem_str = to_lower_case(one_sentence_clean(problem_str))
            problem_str = twogroups_custom_sub("(\d+)krát", "~1 krát", problem_str)
            problem_str, nums, idxs = replace_nums(problem_str)

            # equation = to_lower_case(re.sub("=[0-9]+", " = x", equation))
            equation = remove_equals(equation, prefix=True)
            equation = replace_nums_equation(equation, nums)

            # todo solution type ceil division etc. v svamp paperu se bere numericke hodnoceni bez tech r10 etc picovinek

            equation = " ".join(infixToPrefix(equation.split()))

            res = re.search(NUMBER_PATTERN, answer)
            answer = float(res.group(0))

            eval_eq(equation, nums, answer)

            problem = (problem_str, nums, equation, answer)
            problem_list.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")
    return problem_list


def clear_math23knum_fun(num):
    num = num.group(0)
    num = re.sub("\s+", "", num)
    num = num[1:-1]
    num = num.replace("(", "")
    num = num.replace(")", "")
    num = f"({num})"
    return num


def repair_Math23ktext(text):
    pattern = re.compile("\(\d+\)\s*/\s*\(\d+\)")
    text = re.sub(pattern, clear_math23knum_fun, text)
    return text


def convert_percent_fun(mtch):
    if mtch.group(1):
        num = float(mtch.group(1))
    else:
        num = float(mtch.group(2))
    num /= 100
    return str(num)


def clean_Math23k_equation(eq, problem_numbers, g_numbers):
    """
    converts percentages to their multiplicative values
    replaces N{num} with their numeric values
    checks if equation doesn't contain unknown symbols or numbers not in the text or specific constants
    :param eq:
    :param problem_numbers:
    :param g_numbers:
    :return:
    """
    en_eq = replace_numbers(eq, pattern="N(\d+)", numbers=problem_numbers)
    numbers = re.findall("\(\d+/\d+\)|\d+\.\d+%?|\d+%?", en_eq)
    en_eq = re.sub("(\d+\.\d+)%|(\d+)%", convert_percent_fun, en_eq) #convert percentages
    invalid_chars = re.findall("[^+*\-/=()\s.]", en_eq)
    for c in invalid_chars:  #some non-operator, non-numeric character appears equation?
        if not is_number(c):
            return None
    for n in numbers:        #number in equation isn't in problem numbers or generated numbers
        if n not in problem_numbers and n not in g_numbers:
            return None
    return en_eq


def Math23k_problem_list(path):
    print("\nWorking on Math23k data_and_data_processing...")
    not_translated_indices = set()
    no_en_text_counter = 0
    no_text_both = 0
    eqs_file = get_abs_project_path() + "/stuff/data_and_data_processing/datasets/Math23K/Math_23K_better_ed.json"
    #repair the suipplied json with better tagged numbers and equations
    with open(eqs_file, encoding='utf-8-sig') as fh:
        dtn = fh.read()
        dtn = dtn.replace("}", "},")
        dtn = dtn[:-3] + dtn[-2:]
        eqs_data = json.loads(dtn)
    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)
    for i in range(len(json_data)):
        d_mine = json_data[i]
        d_eqs = eqs_data[i]
        assert d_mine["id"] == d_eqs["id"]  # sanity check
        if not d_mine["en_text"]:
            no_en_text_counter += 1
            not_translated_indices.add(i)
            if not d_eqs["segmented_text"]:
                no_text_both += 1
        problem_str_en = d_mine["en_text"]  # change the equation to the one provided in Graph2Tree data
        d_mine["en_text"] = to_lower_case(en_number_tag(repair_Math23ktext(problem_str_en))) #correctly tag composite numbers, translate en numbers into digits
        d_mine["segmented_text"] = d_eqs["segmented_text"] #use the better chinese text with correctly expressed fraction numbers

        def correct_brackets(eq):
            eq = eq.replace("[", "(")
            eq = eq.replace("]", ")")
            return eq

        d_mine["equation"] = correct_brackets(d_eqs["equation"])
        d_mine["ans"] = d_eqs["ans"]
    if not_translated_indices:
        save_file(not_translated_indices, "../cache/non_translated_chinese_problems.p")
        print("non all chinese data are translated!")
        sys.exit(1)
    pairs_en, temp_g_en, copy_nums_en = transfer_num_MATH23K(json_data, "en_text")
    pairs_ch, temp_g_ch, copy_nums_ch = transfer_num_MATH23K(json_data, "segmented_text")
    #temp_g_en = temp_g_ch = ['1', '3.14'] #also, there's a large number of untagged numbers in the english text apparently, cause? #todo - could filter those two so it would pass better with other data
    temp_g_en = temp_g_ch = [] #for consistency with the english data, we chose to discard problems with new equation numbers
    invalid_en_count = 0

    # validate
    valid_pairs = []
    for i in range(len(pairs_en)):
        ans = re.sub("(\d+\.\d+)%|(\d+)%", convert_percent_fun, eqs_data[i]['ans']) #i could easily convert the percentage values to their multiplicative amounts since all of them are explicitly mentioned, would not work for multiplicative situations
        try:
            ans = float(ans)
        except:
            try:
                ans = eval(ans)
            except:
                # print(f"wrong answer {ans}")
                continue #todo salvage specifics examples

        en = pairs_en[i]
        ch = pairs_ch[i]
        ch_eq = " ".join(ch[1])
        ch_eq_cleaned = clean_Math23k_equation(ch_eq, ch[2], temp_g_ch)
        if not ch_eq_cleaned:
            # print(f"wrong ch eq {ch_eq}, i: {i}")
            continue
        en_eq = " ".join(en[1])
        en_eq_cleaned = clean_Math23k_equation(en_eq, en[2], temp_g_en)

        if en_eq_cleaned and ch_eq_cleaned and len(en_eq) != len(ch_eq):
            print()
        if not en_eq_cleaned:
            # print(f"wrong en eq {en_eq}, i: {i}")
            invalid_en_count += 1
            continue
            # assert en_eq_cleaned
        en_val = eval(en_eq_cleaned)
        ch_val = eval(ch_eq_cleaned)
        assert en_val == ch_val
        # if en_eq != ch_eq:
        #    print(f"ch: {ch_eq}, en: {en_eq}")
        if not math.isclose(ans, en_val, rel_tol=1e-2, abs_tol=1e-10):
            continue
        valid_pairs.append((en, ch, ans, i))
    print(invalid_en_count)

    data_list_EN = []
    data_list_CH = []
    saved_data = []
    for valid_pair in valid_pairs:
        def num_replace(arr):
            ret = []
            counter = 0
            for st in arr:
                if st == 'NUM':
                    ret.append(f"number{counter}")
                    counter += 1
                else:
                    ret.append(st)
            return ret

        def num_replace_eq_fun(mtch):
            n = int(mtch.group(1))
            return f"number{n}"

        en_question = " ".join(num_replace(valid_pair[0][0]))
        ch_question = " ".join(num_replace(valid_pair[1][0]))
        en_nums = valid_pair[0][2]  # previously as floats, what about the percents?
        ch_nums = valid_pair[1][2]
        en_equation = " ".join(infixToPrefix(re.sub("N(\d+)", num_replace_eq_fun, " ".join(valid_pair[0][1])).split()))
        ch_equation = " ".join(infixToPrefix(re.sub("N(\d+)", num_replace_eq_fun, " ".join(valid_pair[1][1])).split()))
        ans = valid_pair[2]
        data_list_EN.append((en_question, en_nums, en_equation, ans, valid_pair[3]))
        data_list_CH.append((ch_question, ch_nums, ch_equation, ans, valid_pair[3]))

        saved_data.append(json_data[valid_pair[3]])

    return data_list_EN, data_list_CH




def _convert_unary_minus(equation):
    expr = sym.sympify(equation)
    num = []
    for i, arg in enumerate(expr.args):
        if isinstance(arg, Pow):
            den = arg.args[0]
        else:
            num.append(arg)
    den = _reformat_sympy_expression(-1 * den)
    num = _reformat_sympy_expression(-1 * expr.__class__(*num))
    ret = f"( {num} ) / ( {den} )"
    return ret


def _separate_numbers(sentence):
    return re.sub(NUMBER_PATTERN_MINUS, _separate_str_fun, sentence)


def _merge_thousands(sentence):
    def mrg(mtch):
        return f"{mtch.group(1)}{mtch.group(2)}"

    return re.sub("(\d+)\s(\d+)", mrg, sentence)


def MAWPS_CZ_problem_list(path):
    ones = 0
    pis = 0
    print("\nWorking on MAWPS_CZ data_and_data_processing...")
    problems_with_new_nums = []
    problems_with_unary_zero = []
    problems_sympy_fail_new_nums = []
    problems_sympy_fail = []
    problems_unknownn_syms = []
    problem_list = []
    problem_list_EN = []
    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)
    """
    forbidden_problems = {174, 263, 446,
                          # percentages
                          433, 434, 435, 442, 443,
                          #unary minus
                          471,
                          #composite
                          483,
                          #dollars-cents
                          505, 551,
                          #round division
                          709, 3327,
                          #unknown wrong evaluation
                          523, 755,
                          # with new numbers
                          428,
                          # en problematic
                            968, 1055, 1495 }
    """
    # wrong equation or solution, corrected - 3071, 3124, 3162, 3165, 3253, 3298, 3307, 3340
    # edited - operator minus without whitespace - 444, 523, 650, 663, 682, 690, 700, 725, 805, 821, 1015, 1023, 1032, 1036, 1043,
    # 1121, 1124, 1128, 1138, 1478, 1494, 1495, 1528, 1563, 1568, 1601, 1691, 1697, 1735, 1740, 1751, 1811, 1861, 1873, 1918, 2246, 2247
    # 2248, 2249 - 2444, 3100, 3101, 3103, 3107 + nejaky dalsi
    # 762 - implicit reasoning
    forbidden_problems = {  # 263 -
        263,  # nonsense equation
        505, 551,  # dollars-cents
        3327,  # wrong solution; nonsense question
        428,
        1725,  # wrong solution + integer division
        3330,  # equation-solution requires bk
        899    #english version has bad eval and equation is overcomplicated due to inverse numbers
    }
    eq_sets_counter = 0
    empty_counter = 0
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    counter5 = 0
    counter6 = 0
    counter7 = 0
    counter8 = 0
    counter9 = 0
    for i in range(len(json_data)):
        has_cze = False
        if i in forbidden_problems:
            continue
        data = json_data[i]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            # filter systems of equations
            continue_to_en = False
            if len(data["lEquations"]) > 1 or len(data["lSolutions"]) > 1:
                eq_sets_counter += 1
                continue
            counter1 += 1
            problem_str = data["sQuestion"]
            equation_or = data["lEquations"][0]
            equation_or = re.sub("(?<=[\da-zA-Z])-(?=[\da-zA-Z])", " - ", equation_or)
            equation = equation_or
            answer = data["lSolutions"][0]
            answer = answer if isinstance(answer, float) else float(re.search(NUMBER_PATTERN, answer).group(0))
            problem_str = word2number_cs(problem_str)
            problem_str = _merge_thousands(problem_str)
            problem_str = to_lower_case(one_sentence_clean(problem_str))
            problem_str = _separate_numbers(problem_str)

            problem_str, nums, idxs = replace_nums(problem_str)

            # equation = to_lower_case(re.sub("=[0-9]+", " = x", equation))

            nums_valid = has_new_nums(equation, nums)
            counter2 += 1
            if not nums_valid:
                problems_with_new_nums.append(i)
                continue_to_en = True
            if not continue_to_en:
                counter3 += 1
                # these make problems in variable names
                equation = equation.replace("_", "")
                equation = replace_nums_equation(equation, nums)
                equation = correct_eq(equation)
                counter4 += 1
                if not equation:
                    problems_sympy_fail.append(i)
                    continue_to_en = True
            if not continue_to_en:
                counter5 += 1
                unknown_nums = re.findall("(?<!number)\d+", equation)
                if unknown_nums:
                    problems_sympy_fail_new_nums.append(i)
                    if '1' in unknown_nums:
                        ones += 1
                    continue_to_en = True
            if not continue_to_en:
                counter6 += 1
                unknown_syms = re.findall("[a-zA-Z]+|(?<!number)\d+", equation)
                if not all([unknown_sym == 'number' for unknown_sym in unknown_syms]):
                    problems_unknownn_syms.append(i)
                    continue_to_en = True
            # else:
            #    equation = remove_equals(equation, prefix=True)
            #    equation = replace_nums_equation(equation, nums)
            # todo solution type ceil division etc. v svamp paperu se bere numericke hodnoceni bez tech r10 etc picovinek
            if not continue_to_en:
                counter7 += 1
                equation_trimmed = equation.lstrip()
                if equation_trimmed[0] == "-":
                    equation = _convert_unary_minus(equation_trimmed)
                    problems_with_unary_zero.append(i)
                #                continue
                equation = " ".join(infixToPrefix(equation.split()))
                counter8 += 1
                eval_eq(equation, nums, answer)
                counter9 += 1
                problem = (problem_str, nums, equation, answer)
                has_cze = True
                problem_list.append(problem)

            problem_str_en = data["sQuestionEN"]
            #problem_str_en = cz2en_delimiter(problem_str_en) #these are contradictory, differentiable only by context
            problem_str_en = twogroups_custom_sub("(\d+),(\d+)", "~1~2", problem_str_en)
            problem_str_en = _separate_numbers(to_lower_case(one_sentence_clean(problem_str_en)))
            problem_str_en = re.sub("twice(?=\s)"," 2 times ", problem_str_en, re.IGNORECASE)

            problem_str_en, nums_en, idxs_en = replace_nums(problem_str_en)
            equation_en = equation_or

            nums_valid = has_new_nums(equation_en, nums_en)
            if not nums_valid:
                # problems_with_new_nums.append(i)
                continue

            equation_en = equation_en.replace("_", "")
            equation_en = replace_nums_equation(equation_en, nums_en)
            equation_en = correct_eq(equation_en)
            unknown_nums = re.findall("(?<!number)\d+", equation_en)
            unknown_syms = re.findall("[a-zA-Z]+|(?<!number)\d+", equation_en)
            if unknown_nums or not all([unknown_sym == 'number' for unknown_sym in unknown_syms]):
                continue
            equation_trimmed = equation_en.lstrip()
            if equation_trimmed[0] == "-":
                equation_en = _convert_unary_minus(equation_trimmed)
            equation_en = " ".join(infixToPrefix(equation_en.split()))
            eval_eq(equation_en, nums_en, answer)
            check_en(nums_en, equation_or)
            problem_list_EN.append((problem_str_en, nums_en, equation_en, answer))

            """
            ok_list = {543, 815, 917, 968, 1055, 1121, 3908}
            eq_nums = re.findall(NUMBER_PATTERN, data["lEquations"][0])
            eq_nums = [convert_num(n) for n in eq_nums]
            check = check_nums_in(nums_en, eq_nums)
            assert check == -1

            if i not in ok_list:
                for n_cz in nums:
                    ok = False
                    for n_en in nums_en:
                        if are_same(n_cz, n_en):
                            ok = True
                            break
                    assert ok
            """
        else:
            empty_counter += 1

    plen = len(problem_list)
    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")
    problem_list_filtered, cz_dupls = filter_duplicates(problem_list, ret_duplicates=True)
    problem_list_EN, en_dupls = filter_duplicates(problem_list_EN, ret_duplicates=True)
    print(f"-> Filtered {plen - len(problem_list_filtered)} duplicates, f"
          f"inally retrieved {len(problem_list_filtered)} MaWPS problems.")
    return problem_list_filtered, problem_list_EN


def SVAMPCZ_problem_list(path):
    #680 - wrong solution
    print("\nWorking on cs SVAMP data_and_data_processing...")

    forbidden_problems = {50}

    interesting_problems = {

    }

    problem_list = []
    problem_list_EN = []
    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)

    for i in range(len(json_data)):
        problem_n = i + 1
        data = json_data[i]
        # do not include nonaritmetic examples
        forbidden = False
        if problem_n in forbidden_problems:
            forbidden = True
        if forbidden:
            continue

        problem_str = data["Body"] + " " + data["Question"]
        equation = data["Equation"]
        answer = data["Answer"]

        problem_str = word2number_cs(problem_str)
        problem_str = to_lower_case(one_sentence_clean(problem_str))
        problem_str, nums, idxs = replace_nums(problem_str)

        nums_valid = has_new_nums(equation, nums)
        if not nums_valid:
            continue
        equation = replace_nums_equation(equation, nums)
        equation = correct_eq(equation)
        if not equation:
            continue
        unknown_nums = re.findall("(?<!number)\d+", equation)
        if unknown_nums:
            continue
        # else:
        #    equation = remove_equals(equation, prefix=True)
        #    equation = replace_nums_equation(equation, nums)
        # todo solution type ceil division etc. v svamp paperu se bere numericke hodnoceni bez tech r10 etc picovinek
        equation_trimmed = equation.lstrip()
        if equation_trimmed[0] == "-":
            continue
        equation = " ".join(infixToPrefix(equation.split()))
        answer = answer if isinstance(answer, float) else float(re.search(NUMBER_PATTERN, answer).group(0))
        eval_eq(equation, nums, answer)
        problem = (problem_str, nums, equation, answer)
        problem_list.append(problem)

        problem_str_en = data["BodyEN"] + " " + data["QuestionEN"]
        equation_en = data["Equation"]
        problem_str_en = to_lower_case(one_sentence_clean(problem_str_en))
        problem_str_en, nums_en, idxs_en = replace_nums(problem_str_en)
        equation_en = replace_nums_equation(equation_en, nums_en)
        equation_en = " ".join(infixToPrefix(equation_en.split()))
        equation_en = correct_eq(equation_en)
        eval_eq(equation_en, nums_en, answer)
        problem_list_EN.append((problem_str_en, nums_en, equation_en, answer))
    problem_list = filter_duplicates(problem_list)
    problem_list_EN = filter_duplicates(problem_list_EN)
    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")
    print(f"-> Retrieved {len(problem_list_EN)} / {len(json_data)} EN problems.")
    return problem_list, problem_list_EN


def check_en(nums_en, equation):
    eq_nums = re.findall(NUMBER_PATTERN_MINUS, equation)
    eq_nums = [convert_num(n) for n in eq_nums]
    check = check_nums_in(nums_en, eq_nums)
    assert check == -1


def check_nums_in(which_nums, where_should_be):
    for ref_num in where_should_be:
        ok = False
        for tested_num in which_nums:
            if are_same(tested_num, ref_num):
                ok = True
                break
        if not ok:
            return tested_num
    else:
        return -1


def _find_bad_adds(expr, adds):
    if isinstance(expr, sym.Add):
        st = str(expr)
        if st[0] == '-':
            adds.append(expr)
    for arg in expr.args:
        adds = _find_bad_adds(arg, adds)
    return adds


def has_new_nums(equation, text_nums):
    nums_in_eq = re.findall(NUMBER_PATTERN_MINUS, equation)
    ok = True
    for eq_num in nums_in_eq:
        part_ok = False
        for num_in_text in text_nums:
            if math.isclose(float(eq_num), num_in_text,
                            rel_tol=1e-14):
                part_ok = True
                break
        if not part_ok:
            ok = False
            break
    return ok


def _correct_add(add):
    ispos = lambda x: x.as_coeff_Mul()[0].is_positive
    pos, neg = sym.sift(sym.Add.make_args(add), ispos, binary=True)
    pos_final = ' + '.join([str(po) for po in pos])
    neg_final = ' - '.join([str(ne)[1:] for ne in neg])
    if neg_final:
        res = pos_final + ' - ' + neg_final
    else:
        res = pos_final

    return res


def _shift_fnc(objmatch):
    res = objmatch.group(0)
    return res + " "


def _replace_multnums(o_eq):
    eq = re.sub(" ", "", o_eq)
    # find powers
    pows = re.finditer("number(\d+)\*\*(\d+)", eq)
    for pow in pows:
        ostr = pow.group(0)
        affected = pow.group(1)
        pow_v = int(pow.group(2))
        new_str = ['*' if i % 2 == 1 else f"number{affected}" for i in range(pow_v * 2 - 1)]
        new_str = "".join(new_str)
        eq = eq.replace(ostr, new_str)

    eq = re.sub("number\d+", _shift_fnc, eq)
    multnums = re.findall("\d+\*number\d+", eq)
    for mn in multnums:
        pts = mn.split('*')
        eq = eq.replace(mn, f"multnum&{pts[0]}&{pts[1]}")
    eq = re.sub(" ", "", eq)
    parts = re.split('([+*/\-()])', eq)
    new_expr = []
    for part in parts:
        i = part.split('&')
        if i and i[0] == "multnum":
            times = int(i[1])
            var = i[2]
            new_part = ['+' if i % 2 == 1 else var for i in range(times * 2 - 1)]
            new_expr += ["(", *new_part, ")"]
        else:
            new_expr.append(part)

    ret = " ".join(new_expr)
    return ret


def _reformat_sympy_expression(res):
    bad_adds = _find_bad_adds(res, [])
    s = str(res)
    for b_add in bad_adds:
        s_b_add = str(b_add)
        idx = s.find(s_b_add)
        if idx == -1:
            print("warning - bad add not found in the whole problem")
        c_add = _correct_add(b_add)
        s = s.replace(s_b_add, c_add)
    s = _replace_multnums(s)

    return s


def correct_eq(eq):
    eq_split = eq.split()
    nums = []
    lhs = []
    rhs = []
    var = None
    is_pre_equals = True
    for symbol in eq_split:
        if symbol == '=':
            is_pre_equals = False
        elif symbol in OPERATORS:
            if is_pre_equals:
                lhs.append(symbol)
            else:
                rhs.append(symbol)
        elif re.findall("number\d+", symbol):
            if is_pre_equals:
                lhs.append(symbol)
            else:
                rhs.append(symbol)
            nums.append(symbol)
        else:
            # breaks sympify
            if symbol == 'minimum' or symbol == 'fraction' or symbol == 'product' or symbol == 'together':
                symbol = 'X'
            assert var is None or var == symbol
            var = symbol
            if is_pre_equals:
                lhs.append(symbol)
            else:
                rhs.append(symbol)
    # expression already is in the right format
    if var is None:
        return eq
    syms = sym.symbols(nums + [var])
    lh_s = sym.sympify(" ".join(lhs))
    x = " ".join(rhs)
    rh_s = sym.sympify(" ".join(rhs))

    eq_s = Eq(lh_s, rhs=rh_s)

    sol = sym.solve(eq_s, syms[-1])
    if not sol:
        return []
    res = sol[0]

    res = _reformat_sympy_expression(res)

    return res


def eval_eq(eq, nums, answer):
    eq_list = eq.split()
    for j in range(len(eq_list)):
        el = eq_list[j]
        if el[:6] == "number":
            no = int(el[6])
            val = nums[no]
            eq_list[j] = val
    constructed_eq_val = eval_prefix(eq_list)
    assert math.isclose(constructed_eq_val, answer, rel_tol=1e-2)
    try:
        assert math.isclose(constructed_eq_val, answer, rel_tol=1e-2)
    except:
        print("problem inequal to equation evaluation!")


def remove_equals(equation, prefix=False):
    if not prefix:
        try:
            return equation[:equation.index('=')]
        except:
            return equation
    else:
        try:
            return equation[equation.index('=') + 1:]
        except:
            return equation


def replace_nums(st):
    spl = st.split()
    ret_st = []
    nums = []
    idxs = []
    counter = 0
    for i, w in enumerate(spl):
        if is_number(w):
            ret_st.append(f"number{counter}")
            idxs.append(i)
            num = float(w)
            if num.is_integer():
                num = int(num)
            nums.append(num)
            counter += 1
        else:
            ret_st.append(w)

    ret_st = " ".join(ret_st)
    nums = [float(num) for num in nums]
    return ret_st, nums, idxs


def get_type(symbol):
    if symbol == "-":
        return "-"
    if symbol in OPERATORS | {'='}:
        return "op"
    elif is_number(symbol):
        return "number"
    else:
        return "var"


def convert_num(n):
    n = float(n)
    if n.is_integer():
        n = int(n)
    return n


def are_same(n1, n2):
    return math.isclose(n1, n2,
                        rel_tol=1e-14)


def find_close(n, nums, already_mapped):
    first = -1
    for idx, num in enumerate(nums):
        # diff = abs(w-num)

        if math.isclose(n, num,
                        rel_tol=1e-14):  # this can happen due to rounding errors of numbers with a lot of decimal values in text
            first = idx
            if first not in already_mapped:
                already_mapped.add(idx)
                return idx
    if first != -1:
        return first
    else:
        return -1


def del_numbers(matchobj):
    res = re.sub(NUMBER_PATTERN, "", matchobj.group(0))
    return res


def replace_nums_equation(eq, nums):
    eq = re.sub(f"[a-zA-Z_]\d+[a-zA-Z_]?|[a-zA-Z_]?\d+[a-zA-Z_]", del_numbers, eq)

    numbers = re.finditer(NUMBER_PATTERN_MINUS, eq)
    ops = re.finditer("[+*/()=]", eq)
    minus = re.finditer("-(?=\D)",
                        eq)  # anything except digit; - before digit is approached as negative number instead of as operator
    vars = re.finditer("[a-zA-Z_]+", eq)

    seq = []

    for num in numbers:
        seq.append((num.span(), num.group(0), num, "num"))
    for num in ops:
        seq.append((num.span(), num.group(0), num, "op"))
    for num in minus:
        seq.append((num.span(), num.group(0), num, "minus"))
    for num in vars:
        seq.append((num.span(), num.group(0), num, "var"))

    seq = sorted(seq, key=lambda x: x[0][0])

    already_mapped = set()
    ret_new = []
    for s in seq:
        sym = s[1]
        if s[3] == 'num':
            nidx = -1
            n = convert_num(sym)
            if n in nums:
                indices = [i for i, num in enumerate(nums) if n == num]
                mapped_idx = -1
                for idx in indices:
                    if idx not in already_mapped:
                        mapped_idx = idx
                        already_mapped.add(idx)
                        break
                if mapped_idx == -1:
                    mapped_idx = indices[0]
                nidx = mapped_idx
            elif isinstance(n, float):
                nidx = find_close(n, nums, already_mapped)
            if nidx != -1:
                sym = f"number{nidx}"
            else:
                print("warning - number in expression not found in text numbers")
        ret_new.append(sym)
    ret_new = " ".join(ret_new)

    # if ret != ret_new:
    #    print()
    return ret_new


def make_csv(data, path):
    if len(data[0]) == 5:
        columns = ["Question", "Numbers", "Equation", "Answer", "Id"]
    else:
        columns = ["Question", "Numbers", "Equation", "Answer"]
    d_df = []
    for d in data:
        nums = d[1]
        nums = [str(x) for x in nums]
        d_df.append((d[0], " ".join(nums), *d[2:]))
    df = pd.DataFrame(d_df, columns=columns)
    df.to_csv(path, index=False)


def filter_duplicates(data, ret_duplicates=False):
    """

    :param data:
    :param ret_duplicates: if true, returns data *and* duplicates, otherwise only data
    :return:
    """
    duplicates = []
    ret_dupl = []
    for d_tested_idx, d_tested in enumerate(data):
        for i in range(d_tested_idx + 1, len(data)):
            d_c = data[i]
            if d_tested == d_c:
                duplicates.append(i)
                if ret_duplicates:
                    ret_dupl.append(d_c)

    for index in sorted(duplicates, reverse=True):
        del data[index]
    if ret_duplicates:
        return data, ret_dupl
    else:
        return data


def save_svampcsv_data(relative_data_path, svampcsv_folder, split, problem_list_method, has_EN=True):
    abs_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    path = abs_project_path + relative_data_path
    if has_EN:
        data, data_EN = problem_list_method(path)
    else:
        data = problem_list_method(path)
    data = filter_duplicates(data)
    random.shuffle(data)
    threshold = math.ceil(len(data) * (1 - split))
    print(f"processed dataset from {path} with {len(data)} problems")
    train = data[:threshold]
    dev = data[threshold:]
    make_csv(train, f"{abs_project_path}{svampcsv_folder}/train.csv")
    make_csv(dev, f"{abs_project_path}{svampcsv_folder}/dev.csv")

    if has_EN:
        data_EN = filter_duplicates(data_EN)
        random.shuffle(data_EN)
        train_en = data_EN[:threshold]
        dev_en = data_EN[threshold:]
        make_csv(train_en, f"{abs_project_path}{svampcsv_folder}EN/train.csv")
        make_csv(dev_en, f"{abs_project_path}{svampcsv_folder}EN/dev.csv")


def get_dataset_list(relative_data_path, problem_list_method, has_EN=True):
    abs_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = abs_project_path + relative_data_path
    if has_EN:
        data, data_EN = problem_list_method(path)
    else:
        data = problem_list_method(path)
        data_EN = None
    data = filter_duplicates(data)
    if data_EN:
        data_EN = filter_duplicates(data_EN)
    return data, data_EN


def save_svampcsv_data_kfold(svampcsv_folder, data, data_name, data_EN=None,):
    abs_project_path = get_abs_project_path()
    kf = KFold(n_splits=5, shuffle=True)

    for idx, (train_idxs, test_idxs) in enumerate(kf.split(data)):
        train = [data[x] for x in train_idxs]
        test = [data[x] for x in test_idxs]
        os.makedirs(f"{abs_project_path}{svampcsv_folder}{idx}", exist_ok=True)
        os.makedirs(f"{abs_project_path}/stuff/gts/data/svamp_paper_csvs/{data_name}/fold{idx}", exist_ok=True)
        make_csv(train, f"{abs_project_path}{svampcsv_folder}{idx}/train.csv")
        make_csv(test, f"{abs_project_path}{svampcsv_folder}{idx}/dev.csv")
        make_csv(train, f"{abs_project_path}/stuff/gts/data/svamp_paper_csvs/{data_name}/fold{idx}/train.csv")
        make_csv(test, f"{abs_project_path}/stuff/gts/data/svamp_paper_csvs/{data_name}/fold{idx}/dev.csv")

    if data_EN:
        kf = KFold(n_splits=5, shuffle=True)
        for idx, (train_idxs, test_idxs) in enumerate(kf.split(data_EN)):
            train = [data_EN[x] for x in train_idxs]
            test = [data_EN[x] for x in test_idxs]
            os.makedirs(f"{abs_project_path}{svampcsv_folder}EN{idx}", exist_ok=True)
            os.makedirs(f"{abs_project_path}/stuff/gts/data/svamp_paper_csvs/{data_name}/foldEN{idx}/", exist_ok=True)
            make_csv(train, f"{abs_project_path}{svampcsv_folder}EN{idx}/train.csv")
            make_csv(test, f"{abs_project_path}{svampcsv_folder}EN{idx}/dev.csv")
            make_csv(train, f"{abs_project_path}/stuff/gts/data/svamp_paper_csvs/{data_name}/foldEN{idx}/train.csv")
            make_csv(test, f"{abs_project_path}/stuff/gts/data/svamp_paper_csvs/{data_name}/foldEN{idx}/dev.csv")


def get_asdiva_data():
    relative_data_path = "/data_and_data_processing/datasets/datasets_cz_new/cs_ASDiv-a_formatted.json"
    return get_dataset_list(relative_data_path, ASDivACZ_problem_list, has_EN=True)


def get_WP500CZ_data():
    relative_data_path = "/data_and_data_processing/datasets/datasets_cz_new/WP500CZ_ALL.json"
    return get_dataset_list(relative_data_path, WP500CZ_problem_list, has_EN=False)


def get_MAWPSCZ_data():
    relative_data_path = "/data_and_data_processing/datasets/datasets_cz_new/cs_mawps_formatted.json"
    return get_dataset_list(relative_data_path, MAWPS_CZ_problem_list, has_EN=True)


def get_SVAMPCZ_data():
    relative_data_path = "/data_and_data_processing/datasets/datasets_cz_new/cs_svamp_formatted.json"
    return get_dataset_list(relative_data_path, SVAMPCZ_problem_list, has_EN=True)


def get_Math23K_data(helsinky):
    """
    :return: EN a CH data
    """
    if helsinky:
        relative_data_path = "/data_and_data_processing/datasets/Math23K/math23k_trns_helsinki.json"
    else:
        relative_data_path = "/data_and_data_processing/datasets/Math23K/baidu_translate/math23k_baidu_trns_all.json"
    return get_dataset_list(relative_data_path, Math23k_problem_list, has_EN=True)  # has CH actually

def get_Math23K_cs_data():
    """
    :return: EN a CH data
    """
    return get_dataset_list("", Math23k_cs_problem_list, has_EN=True)

def save_cs_asdiva():
    svampcsv_folder = f"{data_folder}/new_asdiv-a_cs/fold"
    data, data_EN = get_asdiva_data()
    save_svampcsv_data_kfold(svampcsv_folder, data, "svamp", data_EN=data_EN)


def save_WP500CZ():
    svampcsv_folder = f"{data_folder}/new_WP500CZ/fold"
    data, _ = get_WP500CZ_data()
    save_svampcsv_data_kfold(svampcsv_folder, data, "wp500cz", data_EN=None)


def save_MAWPSCZ():
    svampcsv_folder = f"{data_folder}/new_mawps_cs/fold"
    data, data_EN = get_MAWPSCZ_data()
    save_svampcsv_data_kfold(svampcsv_folder, data, "mawps", data_EN=data_EN)


def save_SVAMPCZ():
    svampcsv_folder = f"{data_folder}/new_svamp_cs/fold"
    data, data_EN = get_SVAMPCZ_data()
    save_svampcsv_data_kfold(svampcsv_folder, data, "svamp", data_EN=data_EN)


def save_combined_cs():
    svampcsv_folder = f"{data_folder}/new_ASDIVA_p_MAWPS/fold"
    MAWPS_data, MAWPS_data_EN = get_MAWPSCZ_data()
    ASDIVA_data, ASDIVA_data_EN = get_asdiva_data()

    data = MAWPS_data + ASDIVA_data
    data_EN = MAWPS_data_EN + ASDIVA_data_EN
    data = filter_duplicates(data)
    data_EN = filter_duplicates(data_EN)
    # save method shuffles the data

    save_svampcsv_data_kfold(svampcsv_folder, data, "mawps_asdiva", data_EN=data_EN)


def save_Math23k(helsinky):
    if helsinky:
        svampcsv_folder = f"{data_folder}/Math23k_helsinki/fold"
    else:
        svampcsv_folder = f"{data_folder}/Math23k/fold"
    data_EN, data_CH = get_Math23K_data(helsinky)
    save_svampcsv_data_kfold(svampcsv_folder, data_EN, f"math23k_{helsinky}", data_EN=data_CH)

def save_Math23k_cs():
    svampcsv_folder = f"{data_folder}/Math23k_cs/fold"
    data, _ = get_Math23K_cs_data()
    save_svampcsv_data_kfold(svampcsv_folder, data, "math23k_cs", data_EN=None)

def get_Math23k_cs_raw():
    data_EN, _ = get_Math23K_data(helsinky=False)
    cs_problems = open(get_abs_project_path() + "/stuff/data_and_data_processing/datasets/Math23K/cs_problems_.txt", 'r')
    math23k_json = get_math23k_json()
    ret = []
    for d in data_EN:
        cs_text = cs_problems.readline()
        cur = math23k_json[d[4]] #index
        cur["cs_text"] = cs_text
        ret.append(cur)
    return ret

def check_problem_valid_basic(equation, nums, answer):
    """
    checks if the problem passes basic criteria - equation doesn't have any number not tagged in text and the equation
    evaluates correctly
    :param equation:
    :param nums:
    :param answer:
    :return:
    """
    eval_eq(equation, nums, answer)
    return not has_new_nums(equation, nums)

def Math23k_cs_problem_list(dummy):
    Math23k_cs_raw = get_Math23k_cs_raw()
    problem_list = []
    #edit the czech text to transferable format
    for d in Math23k_cs_raw:
        problem_str = d['cs_text']
        problem_str = twogroups_custom_sub("(\d)\s%", "~1%", problem_str)
        problem_str = cz2en_delimiter(problem_str)
        problem_str = one_sentence_clean(problem_str, replace_percent=False)
        problem_str = to_lower_case(problem_str)
        problem_str = word2number_cs(problem_str)
        d['cs_text'] = problem_str

    problems, temp_g_en, copy_nums_en = transfer_num_MATH23K(Math23k_cs_raw, "cs_text")

    new_numbers_counter = 0
    wrong_eval_counter = 0
    for i, problem in enumerate(problems):
        """
        def edit_equation(equation, nums):
            equation = to_lower_case(re.sub("=[0-9]+", " = x", equation))
            equation = remove_equals(equation)
            equation = separate_pattern(equation, "-")
            equation = replace_nums_equation(equation, nums)
            equation = " ".join(infixToPrefix(equation.split()))
            return equation
        """
        def num_replace(arr):
            arr = arr.split()
            ret = []
            counter = 0
            for st in arr:
                if st == 'NUM':
                    ret.append(f"number{counter}")
                    counter += 1
                else:
                    ret.append(st)
            return " ".join(ret)

        problem_str = " ".join(problem[0])
        problem_str = num_replace(problem_str)
        nums = problem[2]
        answer = re.sub("(\d+\.\d+)%|(\d+)%", convert_percent_fun, Math23k_cs_raw[i]['ans'])
        try:
            answer = float(answer)
        except:
                answer = eval(answer)

        equation = " ".join(problem[1])

        substituted_eq = clean_Math23k_equation(equation, nums, [])
        if not substituted_eq:
            new_numbers_counter += 1
            continue
        eq_val = eval(substituted_eq)
        if not math.isclose(answer, eq_val, rel_tol=1e-2, abs_tol=1e-10):
            wrong_eval_counter += 1
            continue

        equation = equation.replace("N", 'number')
        equation = " ".join(infixToPrefix(equation.split()))

        fproblem = (problem_str, nums, equation, answer)
        problem_list.append(fproblem)

        # todo in number converter - add replace for "nulu" - 0
        # třicet, devet /nefunguje na zacatku vet? na začátku věty selhalo taky, dvaadvacet jednadvacet taky
        # petinasobek, desetinasobkem nezna
        # čtyř, dvou, deseti, sedmi, tri
        # jedenáctých
        # čtrnácti


        # equation = "10-(3+2)=5"

        # todo solution type ceil division etc. v svamp paperu se bere numericke hodnoceni bez tech r10 etc picovinek

        #equation = edit_equation(equation, nums)
    return problem_list, None


def save_combined_cs2():
    svampcsv_folder = f"{data_folder}/new_ASDIVA_p_MAWPS_p_SVAMP/fold"
    MAWPS_data, MAWPS_data_EN = get_MAWPSCZ_data()
    ASDIVA_data, ASDIVA_data_EN = get_asdiva_data()
    SVAMP_data, SVAMP_data_EN = get_SVAMPCZ_data()

    data = MAWPS_data + ASDIVA_data + SVAMP_data
    data_EN = MAWPS_data_EN + ASDIVA_data_EN + SVAMP_data_EN
    data, dupls = filter_duplicates(data, ret_duplicates=True)
    data_EN, dupls_EN = filter_duplicates(data_EN, ret_duplicates=True)
    # save method shuffles the data

    save_svampcsv_data_kfold(svampcsv_folder, data, "all_engl", data_EN=data_EN)

def save_all():
    svampcsv_folder = f"{data_folder}/all/fold"
    MAWPS_data, _ = get_MAWPSCZ_data()
    ASDIVA_data, _ = get_asdiva_data()
    SVAMP_data, _ = get_SVAMPCZ_data()
    Math23k_data, _ = get_Math23K_cs_data()
    print()
    data = MAWPS_data + ASDIVA_data + SVAMP_data + Math23k_data
    data, dupls = filter_duplicates(data, ret_duplicates=True)
    # save method shuffles the data

    save_svampcsv_data_kfold(svampcsv_folder, data, "all", data_EN=None)

def save_all_plus_wp500cz():
    svampcsv_folder = f"{data_folder}/all_wp500cz/fold"
    MAWPS_data, _ = get_MAWPSCZ_data()
    ASDIVA_data, _ = get_asdiva_data()
    SVAMP_data, _ = get_SVAMPCZ_data()
    Math23k_data, _ = get_Math23K_cs_data()
    WP500CZ_data, _ = get_WP500CZ_data()

    WP500testidxs = set(np.random.choice(500, 150, replace=False))
    WP500testdata = [WP500CZ_data[i] for i in WP500testidxs]
    WP500traindata = [WP500CZ_data[i] for i in range(500) if i not in WP500testidxs]

    data = MAWPS_data + ASDIVA_data + SVAMP_data + Math23k_data + WP500traindata
    data = filter_duplicates(data)
    # save method shuffles the data

    save_svampcsv_data_kfold(svampcsv_folder, data, "all_wp500cz", data_EN=None)
    save_whole_datalist(f"{data_folder}/all_wp500cz/test", "dev.csv",
                        WP500testdata)


def save_combined_cs3():
    svampcsv_folder = f"{data_folder}/new_mixed_native_nonnative2_cs/fold"
    MAWPS_data, _ = get_MAWPSCZ_data()
    ASDIVA_data, _ = get_asdiva_data()
    SVAMP_data, _ = get_SVAMPCZ_data()
    WP500CZ_data, _ = get_WP500CZ_data()

    WP500testidxs = set(np.random.choice(500, 150, replace=False))
    WP500testdata = [WP500CZ_data[i] for i in WP500testidxs]
    WP500traindata = [WP500CZ_data[i] for i in range(500) if i not in WP500testidxs]

    data = MAWPS_data + ASDIVA_data + WP500traindata
    data = filter_duplicates(data)
    save_svampcsv_data_kfold(svampcsv_folder, data, "mix2", data_EN=None)
    save_whole_datalist(f"{data_folder}/new_mixed_native_nonnative_cs2/test", "dev.csv",
                        WP500testdata)




def save_whole(result_folder, result_file, get_data_method):
    project_path = get_abs_project_path()
    data, data_EN = get_data_method()
    os.makedirs(f"{project_path}{result_folder}", exist_ok=True)
    make_csv(data, f"{project_path}{result_folder}/{result_file}")
    if data_EN is not None:
        make_csv(data_EN, f"{project_path}{result_folder}/{result_file}EN")


def save_whole_datalist(result_folder, result_file, data, data_EN=None):
    project_path = get_abs_project_path()
    os.makedirs(f"{project_path}{result_folder}", exist_ok=True)
    make_csv(data, f"{project_path}{result_folder}/{result_file}")
    if data_EN is not None:
        make_csv(data_EN, f"{project_path}{result_folder}/{result_file}EN")


def save_WP500CZWHOLE():
    save_whole(f"{data_folder}/new_WP500CZ/whole", "dev.csv", get_WP500CZ_data)

"""
def test():
    #get_WP500CZ_data()
    get_asdiva_data()
    #get_SVAMPCZ_data()
    #get_MAWPSCZ_data()

    # asdiva_list, asdiva_list_EN = get_asdiva_data()
    # abs_path = get_abs_project_path()
    # data_rel_path = "/data_and_data_processing/datasets/Math23K/baidu_translate/old/math23k_trns.json"
    # Math23k_problem_list(abs_path + data_rel_path)

    print()
"""
def create_data():
    save_cs_asdiva()
    save_MAWPSCZ()
    save_SVAMPCZ()
    save_Math23k(helsinky=False)
    save_Math23k_cs()


if __name__ == "__main__":
    pass
    #save_combined_cs3()
    #get_Math23K_data(helsinky=False)
    # save_Math23k_cs()
    #save_all()
    #save_all_plus_wp500cz()
    # test_create_tree()
    # split = 0.2
    # save_WP500CZWHOLE()
    # save_combined_cs()
    # save_combined_cs3()
    # save_Math23k()
    # i can't reshuffle the results again, especially bc test set leakage
    # save_WP500CZ()
    #save_MAWPSCZ()
    #save_cs_asdiva()
    #save_SVAMPCZ()
    #save_Math23k(helsinky=False)

    #test()
