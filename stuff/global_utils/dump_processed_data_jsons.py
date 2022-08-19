import glob
import os

from stuff.global_utils.general import get_json_from_processed_data, dump_dataset
from stuff.data_and_data_processing.data_loading_wrapper import load_data_from_griffiths_binary


def dump_processed_data_jsons(abs_path):
    filepath_len = len(abs_path)
    files = glob.glob(f"{abs_path}/*")
    for file in files:
        path = os.path.abspath(file)
        if not os.path.isdir(path):
            dat = load_data_from_griffiths_binary(os.path.abspath(file))
            dat = get_json_from_processed_data(dat)
            dump_dataset(dat, f"{abs_path}/json_dump/{file[filepath_len:-2]}.json")


