import fire
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from tqdm import tqdm

from audio import *
from hparams import hparams as hp


def convert_file(path):
    y = load_wav(path)
    mel = melspectrogram(y).T
    return mel.astype(np.float32)


def _process_utterance(path, mel_dir):
    directory = os.path.split(path)[: -1]
    fid = os.path.split(path)[-1].split('.')[0]
    m = convert_file(path)
    np.save(f'{mel_dir}/{fid}.npy', m)
    return fid


def main(wav_dir, mel_dir):
    os.makedirs(mel_dir, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []
    for filename in os.listdir(wav_dir):
        wav_path = os.path.join(wav_dir, filename)
        futures.append(executor.submit(partial(
            _process_utterance, wav_path, mel_dir)))
    results = [future.result() for future in tqdm(futures)]


if __name__ == '__main__':
  fire.Fire(main)
  print('Completed.')
