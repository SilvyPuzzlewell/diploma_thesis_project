from __future__ import division, unicode_literals, print_function
import re
from .core import NumberParseException, placevalue
from decimal import Decimal, localcontext


VOCAB = {
    'nula': (0, 'Z'),
    'nulu': (0, 'Z'),
    #'oh': (0, 'Z'),
    #'a': (1, 'A'),
    'jedna': (1, 'D'),
    'jeden': (1, 'D'),
    'dvě': (2, 'D'),
    'dva': (2, 'D'),
    'tři': (3, 'D'),
    'čtyři': (4, 'D'),
    'pět': (5, 'D'),
    'šest': (6, 'D'),
    'sedm': (7, 'D'),
    'osm': (8, 'D'),
    'devět': (9, 'D'),
    'deset': (10, 'M'),
    'jedenácti': (11, 'M'),
    'dvanácti': (12, 'M'),
    'třinácti': (13, 'M'),
    'čtrnácti': (14, 'M'),
    'patnácti': (15, 'M'),
    'šestnácti': (16, 'M'),
    'sedmnácti': (17, 'M'),
    'osmnácti': (18, 'M'),
    'devatenácti': (19, 'M'),
    'dvaceti': (20, 'T'),
    'třiceti': (30, 'T'),
    'čtyřiceti': (40, 'T'),
    'padesáti': (50, 'T'),
    'šedesáti': (60, 'T'),
    'sedmdesáti': (70, 'T'),
    'osmdesáti': (80, 'T'),
    'devadesáti': (90, 'T'),
    'jedenáct': (11, 'M'),
    'dvanáct': (12, 'M'),
    'třináct': (13, 'M'),
    'čtrnáct': (14, 'M'),
    'patnáct': (15, 'M'),
    'šestnáct': (16, 'M'),
    'sedmnáct': (17, 'M'),
    'osmnáct': (18, 'M'),
    'devatenáct': (19, 'M'),
    'dvacet': (20, 'T'),
    'třicet': (30, 'T'),
    'čtyřicet': (40, 'T'),
    'padesát': (50, 'T'),
    'šedesát': (60, 'T'),
    'sedmdesát': (70, 'T'),
    'osmdesát': (80, 'T'),
    'devadesát': (90, 'T'),
    'sto': (100, 'H'),
    'sta': (100, 'H'),
    'stě': (100, 'H'),
    'tisíc': (10**3, 'X'),
    'tisíce': (10**3, 'X')

}

"""
'million': (10**6, 'X'),
'billion': (10**9, 'X'),
'trillion': (10**12, 'X'),
'quadrillion': (10**15, 'X'),
'quintillion': (10**18, 'X'),
'sextillion': (10**21, 'X'),
'septillion': (10**24, 'X'),
'octillion': (10**27, 'X'),
'nonillion': (10**30, 'X'),
'decillion': (10**33, 'X'),
'undecillion': (10**36, 'X'),
'duodecillion': (10**39, 'X'),
'tredecillion': (10**42, 'X'),
'quattuordecillion': (10**45, 'X'),
'quindecillion': (10**48, 'X'),
'sexdecillion': (10**51, 'X'),
'septendecillion': (10**54, 'X'),
'octodecillion': (10**57, 'X'),
'novemdecillion': (10**60, 'X'),
'vigintillion': (10**63, 'X'),
'centillion': (10**303, 'X')
"""


class FST:
    def __init__(self):
        def f_zero(self, n):
            assert n == 0
            self.value = n

        def f_add(self, n):
            self.value += n

        def f_mul(self, n):
            output = self.value * n
            self.value = 0
            return output

        def f_mul_hundred(self, n):
            assert n == 100
            self.value *= n

        def f_mul_hundred_and_add(self, n):
            self.value *= 100
            self.value += n

        def f_ret(self, _):
            return self.value

        self.value = 0
        self.state = 'S'
        # self.states = {'S', 'D', 'T', 'M', 'H', 'X', 'Z', 'A', 'F'}
        self.edges = {
            ('S', 'Z'): f_zero,    # 0
            ('S', 'D'): f_add,     # 9
            ('S', 'T'): f_add,     # 90
            ('S', 'M'): f_add,     # 19
            #new--
            ('S', 'H'): f_add,
            ('S', 'X'): f_add,
            #--
            ('S', 'A'): f_add,    # 100
            ('S', 'F'): f_ret,     # 1
            ('D', 'H'): f_mul_hundred,     # 900
            ('D', 'X'): f_mul,     # 9000
            ('D', 'F'): f_ret,     # 9
            ('T', 'D'): f_add,     # 99
            ('D', 'T'): f_mul_hundred_and_add,     # 990 (nine ninety)
            ('D', 'M'): f_mul_hundred_and_add,     # 919 (nine nineteen)
            ('T', 'H'): f_mul_hundred,
            ('T', 'X'): f_mul,     # 90000
            ('T', 'F'): f_ret,     # 90
            ('M', 'H'): f_mul_hundred,
            ('M', 'X'): f_mul,     # 19000
            ('M', 'F'): f_ret,     # 19
            ('H', 'D'): f_add,     # 909
            ('H', 'T'): f_add,     # 990
            ('H', 'M'): f_add,     # 919
            ('H', 'X'): f_mul,     # 900000
            ('H', 'F'): f_ret,     # 900
            ('X', 'D'): f_add,     # 9009
            ('X', 'T'): f_add,     # 9090
            ('X', 'M'): f_add,     # 9019
            ('X', 'F'): f_ret,     # 9000
            ('Z', 'F'): f_ret,     # 0
            #('A', 'H'): f_mul_hundred,     # 100
            #('A', 'X'): f_mul,      # 1000
            #('A', 'F'): f_ret,      # 1
        }

    def transition(self, token):
        value, label = token
        try:
            edge_fn = self.edges[(self.state, label)]
        except KeyError:
            raise NumberParseException("Invalid number state from "
                                       "{0} to {1}".format(self.state, label))
        self.state = label
        return edge_fn(self, value)


def compute_placevalues(tokens):
    """Compute the placevalues for each token in the list tokens"""
    pvs = []
    for tok in tokens:
        #todo cz - nepouziva se 'point', "tecka"?
        if tok == 'point':
            pvs.append(0)
        else:
            pvs.append(placevalue(VOCAB[tok][0]))
    return pvs

def tokenize(text):
    tokens = re.split(r"[\s,\-]+(?:and)?", text.lower())
    # Remove empty strings caused by split
    tokens = [tok for tok in tokens if tok]
    try:
        # don't use generator here because we want to raise the exception
        # here now if the word is not found in vocabulary (easier debug)
        decimal = False
        parsed_tokens = []
        decimal_tokens = []
        mul_tokens = []
        pvs = compute_placevalues(tokens)
        # Loop until all trailing multiplier tokens are removed and added to mul_tokens; Loop conditions:
        # 1: The last token in the list must have the highest placevalue of any token
        # 2: The list of tokens must be longer than one (to prevent extracting all tokens as mul_tokens)
        # 3: The maximum placevalue must be greater than 1 (This limits our mul_tokens to "hundred" or greater)
        while max(pvs) == pvs[-1] and len(pvs) > 1 and max(pvs) > 1:
            mul_tokens.insert(0, VOCAB[tokens.pop()])
            pvs.pop()

        for token in tokens:
            if token == 'point':
                if decimal:
                    raise ValueError("Invalid decimal word "
                                        "'{0}'".format(token))
                else:
                    decimal = True
            else:
                if decimal:
                    decimal_tokens.append(VOCAB[token])
                else:
                    parsed_tokens.append(VOCAB[token])
    except KeyError as e:
        raise ValueError("Invalid number word: "
                         "{0} in {1}".format(e, text))
    if decimal and not decimal_tokens:
        raise ValueError("Invalid sequence: no tokens following 'point'")
    return parsed_tokens, decimal_tokens, mul_tokens


def compute(tokens):
    """Compute the value of given tokens.
    TODO: memoize placevalue checking at every step
    """
    fst = FST()
    outputs = []
    last_placevalue = None
    for token in tokens:
        out = fst.transition(token)
        if out:
            outputs.append(out)
            if last_placevalue and last_placevalue <= placevalue(outputs[-1]):
                raise NumberParseException("Invalid sequence "
                                           "{0}".format(outputs))
            last_placevalue = placevalue(outputs[-1])
    outputs.append(fst.transition((None, 'F')))
    if last_placevalue and last_placevalue <= placevalue(outputs[-1]):
        raise NumberParseException("Invalid sequence "
                                   "{0}".format(outputs))
    return sum(outputs)

def compute_multipliers(tokens):
    """
    Determine the multiplier based on the tokens at the end of
    a number (e.g. million from "one thousand five hundred million")
    """
    total = 1
    for token in tokens:
        value, label = token
        total *= value
    return total


def compute_decimal(tokens):
    """Compute value of decimal tokens."""
    with localcontext() as ctx:
        # Locally sets decimal precision to 15 for all computations
        ctx.prec = 15
        total = Decimal()
        place = -1
        for token in tokens:
            value, label = token
            if label not in ('D', 'Z'):
                raise NumberParseException("Invalid sequence after decimal "
                                           "point")
            else:
                total += value * Decimal(10) ** Decimal(place)
                place -= 1
    return float(total) if tokens else 0


def evaluate(text):
    tokens, decimal_tokens, mul_tokens = tokenize(text)
    if not tokens and not decimal_tokens:
        raise ValueError("No valid tokens in {0}".format(text))
    return (compute(tokens) + compute_decimal(decimal_tokens)) * compute_multipliers(mul_tokens)

