import re

read_file = open("cs_ASDiv-a.json", 'r')
write_file = open("test.json", 'w')

text = read_file.read()
#ed_text = re.sub("(?<![kK]oli)krát[\s+\.,]", " krát ", text)

single_numbers = {
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
def dashrepl(matchobj):
    composite_number = matchobj.group(0)
    digit = -1
    for numstring, number in single_numbers.items():
        if composite_number.startswith(numstring):
            digit = number
            break
    ten_number_str = composite_number.replace(numstring + 'a', '')
    ten_number = ten_numbers[composite_number.replace(numstring+'a', '')]
    nmb = digit + ten_number
    return str(nmb)


#test_obj  = re.search("(?<![kK]oli)krát[\s+\.,]", text)
pattern = "(jedna|dva|tři|čtyři|pět|šest|sedm|osm|devět)a(dvacet|třicet|čtyřicet|padesát|šedesát|sedmdesát|osmdesát|devadesát)"
test_obj = re.sub(pattern, dashrepl, text)

print("...")
#write_file.write(ed_text)