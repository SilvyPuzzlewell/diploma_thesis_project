from __future__ import absolute_import

import math

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .ProblemPrecisionCalculator import ProblemPrecisionCalculator
import re

from .....global_utils.general import is_number
from .....global_utils.equation_converter_mine import evalPostfix


class Scorer():
    def __init__(self, hypothesis_actual_list):
        self.__hypothesis_list = hypothesis_actual_list

    def average(self, lst):
        return sum(lst) / len(lst)

    def get(self):
        bleu_average = []
        perfect = []
        precision = []

        for hypothesis, actual in self.__hypothesis_list:
            hypothesis = re.sub(r".0", "", hypothesis)
            actual = re.sub(r".0", "", actual)

            pc = ProblemPrecisionCalculator(actual, hypothesis)
            precision.append(pc.get_precision())

            # Format for BLEU
            bleu_hyp = hypothesis.split()
            bleu_act = actual.split()

            min_length = min(len(bleu_act), len(bleu_hyp))

            score = "%1.4f" % sentence_bleu([bleu_act],
                                            bleu_hyp,
                                            weights=(0.5, 0.5),
                                            smoothing_function=SmoothingFunction().method2)

            if score[0] == '1':
                perfect.append((hypothesis, actual))

            bleu_average.append(float(score))

        number_perfect = len(perfect)

        number_of_attempts = len(bleu_average)

        perfection_percentage = (number_perfect / number_of_attempts) * 100

        short_percentage = float("%3.2f" % perfection_percentage)

        avg_precision = (self.average(precision)) * 100

        short_precision = float("%3.2f" % avg_precision)

        bleu = self.average(bleu_average) * 100

        short_bleu = float("%3.2f" % (bleu))

        return number_of_attempts, short_percentage, short_precision, short_bleu


class ScorerDev():
    def __init__(self, hypothesis_actual_list):
        self.__hypothesis_list = hypothesis_actual_list

    def average(self, lst):
        return sum(lst) / len(lst)

    def print_pr(self, pr, hyp, act):
        print(pr)
        print(f"hypothesis: {hyp}")
        print(f"actual: {act}")
    def get(self):
        bleu_average = []
        perfect = []
        precision = []

        num_correct = 0

        for hypothesis, actual, problem in self.__hypothesis_list:
            #self.print_pr(problem, hypothesis, actual)
            hypothesis = re.sub(r"\.\s", " ", hypothesis)
            actual = re.sub(r"\.\s", " ", actual)
            #print("***aft***")
            #self.print_pr(problem, hypothesis, actual)

            pc = ProblemPrecisionCalculator(actual, hypothesis)
            precision.append(pc.get_precision())

            # Format for BLEU
            bleu_hyp = hypothesis.split()
            bleu_act = actual.split()

            min_length = min(len(bleu_act), len(bleu_hyp))

            score = "%1.4f" % sentence_bleu([bleu_act],
                                            bleu_hyp,
                                            weights=(0.5, 0.5),
                                            smoothing_function=SmoothingFunction().method2)
            val_hyp = evalPostfix(hypothesis)
            val_actual = evalPostfix(actual)
            if score[0] == '1':
                perfect.append((hypothesis, actual))
            if is_number(val_hyp):
                val_hyp = float(val_hyp)
                if is_number(val_actual):
                    val_actual = float(evalPostfix(actual))
                #perfect translation

                #if perfect translation *or* numerically close, it c
                    if math.isclose(val_hyp, val_actual, abs_tol=0.005) or score[0] == '1':
                        num_correct += 1

                    if score[0] != '1' and math.isclose(val_hyp, val_actual, abs_tol=0.005):
                        x = 1
                        #print("***same numeric value***")
                        #print(problem)
                        #print(f"hypothesis: {hypothesis}, val: {val_hyp}")
                        #print(f"actual: {actual}, val: {val_actual}")
            #actual value not parsable, this shouldn't happen!
                else:
                    x = 1
                    #print("***label equation unparsable, bad situation, very fucked***")
                    #print(problem)
                    #print(f"hypothesis: {hypothesis}, val: {val_hyp}")
                    #print(f"actual: {actual}, val: {val_actual}")
            #hypothesis not parsable
            else:
                x = 1
                #print("***hypothesis unparsable***")
                #print(problem)
                #print(f"hypothesis: {hypothesis}, val: {val_hyp}")
                #print(f"actual: {actual}, val: {val_actual}")

            bleu_average.append(float(score))

        number_perfect = len(perfect)

        number_of_attempts = len(bleu_average)

        perfection_percentage = (number_perfect / number_of_attempts) * 100
        correct_percentage = (num_correct / number_of_attempts) * 100

        short_percentage = float("%3.2f" % perfection_percentage)
        correct_percentage = float("%3.2f" % correct_percentage)

        avg_precision = (self.average(precision)) * 100

        short_precision = float("%3.2f" % avg_precision)

        bleu = self.average(bleu_average) * 100

        short_bleu = float("%3.2f" % (bleu))

        return number_of_attempts, short_percentage, short_precision, short_bleu, correct_percentage
