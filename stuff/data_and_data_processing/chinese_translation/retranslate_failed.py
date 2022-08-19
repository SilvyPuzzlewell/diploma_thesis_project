import sys
sys.path.append('../../../')
from stuff.global_utils.utils import load_file, save_json
from chinese_trns_script import get_baidu_translation
from stuff.data_and_data_processing.data_loading_wrapper import get_math23k_json, get_translated_math23k_json, MATH23K_TRANSLATED_FILE

original_data = get_math23k_json()
translated_data = get_translated_math23k_json()
failed_translation_idxs = load_file("/home/sylvi/diplomka/my_code/cache/non_translated_chinese_problems.p")

for c, idx in enumerate(failed_translation_idxs):
    ch_text = original_data[idx]["segmented_text"]
    en_text = ""
    counter = 0
    while not en_text:
        en_text = get_baidu_translation(ch_text)
        if not en_text:
            print(f"translation {ch_text} failed {counter} times!")
            counter += 1
            if counter > 100: # save at least what was already translated
                break
    translated_data[idx]["en_text"] = en_text
    print(f"processed {c} problems out of {len(failed_translation_idxs)}")

save_json(translated_data, MATH23K_TRANSLATED_FILE)