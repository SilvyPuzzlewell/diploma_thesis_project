import re
NUMBER_PATTERN = re.compile("\d+\.\d+|\d+")
NUMBER_PATTERN_MINUS = re.compile("-?\d+\.\d+|-?\d+")
OPERATORS = {'+','-','*','/','(',')'}