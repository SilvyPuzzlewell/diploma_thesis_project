import re
from stuff.data_and_data_processing.word2number_cs import lang_CZ
from stuff.global_utils.utils import cz2en_delimiter

xtimes_dispozition_pattern = re.compile("(?<![kK]oli)kr[á]t[\s+.,]", re.IGNORECASE)
composite_number_pattern = re.compile("(dva|tři|čtyři|pět|šest|sedm|osm|devět)a("
                                      "dvacet|třicet|čtyřicet|padesát|šedesát|sedmdesát|osmdesát|devadesát)i?|"
                                      "(jedna)("
                                      "dvacet|třicet|čtyřicet|padesát|šedesát|sedmdesát|osmdesát|devadesát)i?", re.IGNORECASE)
composite_number_pattern2 = re.compile("(dvacet|třicet|čtyřicet|padesát|šedesát|sedmdesát|osmdesát|devadesát)\s?"
                                       "(jedna|dva|tři|čtyři|pět|šest|sedm|osm|devět)", re.IGNORECASE)
nasobek_pattern = re.compile("(dvoj|dvou|troj|čtyř|pěti|šesti|sedmi|osmi|devíti|deseti|desíti)(?=násob[ek][ku])|"
                             "(dvoj|dvou|troj|čtyř|pěti|šesti|sedmi|osmi|devíti|deseti|desíti)(?=násobkem)", re.IGNORECASE)
pseudozdrobneliny_pattern = re.compile("(dvojk|trojk|čtyřk|pětk|šestk|sedmičk|osmičk|devítk)[ay]", re.IGNORECASE)
additional_pattern = re.compile("dvou|dvoum|třem|(?<=\s)tří(?=\s)|(?<=\s)třech(?=\s)|čtyřem|"
                                "(?<=\s)čtyř(?=\s)|(?<=\s)čtyřech(?=\s)|pěti|šesti|sedmi|osmi|"
                                "devíti|desíti|(?<=\s)deseti(?=\s)", re.IGNORECASE)

additional_numbers = {
"dvou": 2,
"dvoum" : 2,
"třem" : 3,
"tří": 3,
"třech": 3,
"čtyřem" : 4,
"čtyř" : 4,
"čtyřech": 4,
"pěti" : 5,
"šesti" : 6,
"sedmi" : 7,
"osmi" : 8,
"devíti" : 9,
"desíti": 10,
"deseti": 10
}

pseudozdrobneliny_numbers = {
"dvojk" : 2,
"trojk" : 3,
"čtyřk" : 4,
"pětk" : 5,
"šestk" : 6,
"sedmičk" : 7,
"osmičk" : 8,
"devítk" : 9,
}

single_numbers = {
    "nula": 0,
    "jedna":1,
    "dva":  2,
    "tři":  3,
    "čtyři":4,
    "pět":  5,
    "šest": 6,
    "sedm": 7,
    "osm" : 8,
    "devět":9
}

nasobek_numbers = {
    "dvoj": 2,
    "dvou": 2,
    "troj": 3,
    "tří": 3,
    "čtyř": 4,
    "pěti": 5,
    "šesti": 6,
    "sedmi": 7,
    "osmi": 8,
    "devíti" : 9,
    "deseti": 10,
    "desíti" : 10
}


ten_numbers = {
    "dvacet" : 20,
    "třicet" : 30,
    "čtyřicet" : 40,
    "padesát" : 50,
    "šedesát" : 60,
    "sedmdesát" : 70,
    "osmdesát" : 80,
    "devadesát" : 90
}

def composite_number_conversion_fnc(matchobj):
    ten_num_grp = matchobj.group(2)
    if not ten_num_grp:
        ten_num_grp = matchobj.group(4)
    single_number_grp = matchobj.group(1)
    if not single_number_grp:
        single_number_grp = matchobj.group(3)
    return str(ten_numbers[ten_num_grp.lower()] + single_numbers[single_number_grp.lower()])
    """
    global numstring
    composite_number = matchobj.group(1).lower()
    digit = -1
    for numstring, number in single_numbers.items():
        if composite_number.startswith(numstring):
            digit = number
            break
    rest = composite_number.replace(numstring+'a', '')
    if rest[-1] == 'i':
        rest = rest[:-1]
    ten_number = ten_numbers[rest.lower()]
    nmb = digit + ten_number
    return str(nmb)
    """
def composite_number_conversion_fnc2(matchobj):
    ten_num_grp = matchobj.group(1)
    single_number_grp = matchobj.group(2)
    return str(ten_numbers[ten_num_grp.lower()] + single_numbers[single_number_grp.lower()])

def nasobek_fnc(matchobj):
    nasobek = matchobj.group(0)
    return f" {nasobek_numbers[nasobek.lower()]} "

def additional_fnc(matchobj):
    additional = matchobj.group(0)
    return f" {additional_numbers[additional.lower()]} "

def pseudozdrobneliny_fce(matchobj):
    num = matchobj.group(1)
    return f" {pseudozdrobneliny_numbers[num.lower()]} "

def replace_decimal_sep(matchobj):
    decimal_str = matchobj.group(0)
    decimal_str = decimal_str.replace(',', '.')
    return decimal_str


def num_convert_success_wrapper(word):
    try:
        lang_CZ.evaluate(word)
        return True
    except:
        return False

#this is from create_data method in utils, todo move the whole thing there
def cleaning(text):
    # Clean up the data_and_data_processing and separate everything by spaces

    #all dots except acronym dots replaced by " . "
    text = re.sub(r"\d+,\d+", replace_decimal_sep, text)
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)(?<![0-9])(\s+)?\.(\s+)?", " . ",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"(\s+)?\?(\s+)?", " ? ", text)
    #text = re.sub(r",", "", text)
    text = re.sub(r"^\s+", "", text)
    text = text.replace('\n', ' ')
    text = text.replace("'", " '")
    #text = text.replace('%', ' percent')
    text = text.replace('$', ' $ ')
    text = re.sub(r"\.\s+", " . ", text)
    text = re.sub(r"\s+", ' ', text)


    return text


def sentence_word2num(sentence):
    sentence = cleaning(sentence)
    sentence_words = sentence.split(' ')
    last_numbers = []
    last_word_idx = -1

    for word_idx in range(len(sentence_words)-1, -1, -1):
        word = sentence_words[word_idx]
        if num_convert_success_wrapper(word): #try to replace the word
            if not last_numbers:              #?
                last_word_idx = word_idx
            last_numbers.append(word)
        else:
            if last_numbers:                   #means that last number(s) was succ converted, can be inserted into the sequence
                last_numbers.reverse()
                ev_string = ' '.join(last_numbers)
                number = lang_CZ.evaluate(ev_string)
                for i in range(last_word_idx, word_idx, -1):
                    sentence_words.pop(i)
                sentence_words.insert(word_idx+1, str(number))
                last_word_idx = -1
                last_numbers = []
    #in the beginning
    if last_numbers:  # means that last number(s) was succ converted, can be inserted into the sequence
        last_numbers.reverse()
        ev_string = ' '.join(last_numbers)
        number = lang_CZ.evaluate(ev_string)
        for i in range(last_word_idx, word_idx-1, -1):
            sentence_words.pop(i)
        sentence_words.insert(word_idx, str(number))


    ret = ' '.join(sentence_words)
    return ret


def word2number_cs(text_to_convert):
    edited_text = re.sub(xtimes_dispozition_pattern, " krát ", text_to_convert)
    edited_text = sentence_word2num(edited_text)
    edited_text = re.sub(composite_number_pattern, composite_number_conversion_fnc, edited_text)
    edited_text = re.sub(composite_number_pattern2, composite_number_conversion_fnc2, edited_text)
    edited_text = re.sub(nasobek_pattern, nasobek_fnc, edited_text)
    edited_text = re.sub(additional_pattern, additional_fnc, edited_text)
    edited_text = re.sub(pseudozdrobneliny_pattern, pseudozdrobneliny_fce, edited_text)
    edited_text = cz2en_delimiter(edited_text)
    return edited_text


