from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from tqdm import tqdm

import fire
import numpy as np
import os


start_char = u'\u4e00'


def convert_file(filename):
    fid = filename.split('/')[-1].split('.')[0]
    data = np.load(filename)
    unicode = [chr(ord(start_char) + x) for x in data]
    unicode = ''.join(unicode)
    return fid, unicode


def main(dir_in, file_out):
    executor = ProcessPoolExecutor(max_workers=10)
    futures = []
    for filename in os.listdir(dir_in):
        wav_path = os.path.join(dir_in, filename)
        futures.append(executor.submit(partial(
            convert_file, wav_path)))
    results = [future.result() for future in tqdm(futures)]

    with open(file_out, 'w') as fout:
        for fid, unicode in results:
            fout.write('{}|{}\n'.format(fid, unicode))


if __name__ == '__main__':
    fire.Fire(main)
    print('Completed.')