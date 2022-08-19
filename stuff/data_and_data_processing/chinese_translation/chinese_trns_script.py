import json
import subprocess
import huggingface_trns
import threading
from stuff.data_and_data_processing.data_loading_wrapper import get_math23k_json

def get_baidu_translation(trns_text):
    trns = subprocess.run(['node', 'baidu_js_api/test.js', trns_text], stdout=subprocess.PIPE).stdout.decode('utf-8')
    return trns

baidu_translation = get_baidu_translation
helsinki_translation = huggingface_trns.helsinky_translation


def translate(range, trns_function, target_filenam, data):
    cur_problem_list = []
    for i in range:
        cur_problem = data[i]
        txt = cur_problem['segmented_text']
        if not txt:
            continue
        translated_text = ""
        counter = 1
        while not translated_text:
            translated_text = trns_function(txt)
            if not translated_text:
                print(f"translation {txt} failed {counter} times!")
                counter += 1
        cur_problem['en_text'] = translated_text
        cur_problem_list.append(cur_problem)
        print(f"problem num. {i} processed.")

    with open(f"{target_filenam}_helsinki_{range.start}_{range.stop}.json", 'w', encoding='utf-8') as f:
        json.dump(cur_problem_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data = get_math23k_json()

    #parser = argparse.ArgumentParser(description='range')
    #parser.add_argument('--frm', dest='frm', type=int)
    #parser.add_argument('--to', dest='to', type=int)
    #args = parser.parse_args()

    target_filename = "../Math23K/baidu_translate/old/math23k_trns"
    trns_function = helsinki_translation

    #frm = 7000
    #to = 8000

    #frm = args.frm
    #to = args.to
    #print('x')
    #print(frm)
    #print(to)
    #range = range(frm, to)
    threads = []
    #threads += [threading.Thread(target=translate,
    #                            args=(range(i*1000, (i+1)*1000), trns_function, file_name, data))
    #           for i in range(1)]
    threads.append(threading.Thread(target=translate,
                                    args=(range(10000, 23000), trns_function, target_filename, data)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    #translate(range, trns_function, file_name, data)


