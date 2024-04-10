from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

import fire
import html
import numpy as np
import os
import pickle
import re
import sentencepiece as spm


sp = spm.SentencePieceProcessor(model_file='/mnt/public/usr/haohanguo/workspace/vqwordseg/exp/msmcr/aishell3/64/model_32000.model')


def main(script_file, phone_file, dur_file):
    with open(script_file) as fin, open(phone_file, 'w') as fout_p, open(dur_file, 'w') as fout_d:
        for line in fin.readlines():
            fid, unicode = line.split('|')
            proto = sp.encode(unicode, out_type='immutable_proto')
            
            piece_ids_t = [x.id for x in proto.pieces]
            piece_dur_t = [len(x.piece) for x in proto.pieces]

            piece_dur, piece_ids = [], []
            for pid, pdur in zip(piece_ids_t, piece_dur_t):
                if len(piece_dur) == 0 or piece_ids[-1] != pid:
                    piece_dur.append(pdur)
                    piece_ids.append(pid)
                else:
                    piece_dur[-1] += pdur

            fout_p.write("{}|{}\n".format(fid, ' '.join([str(x) for x in piece_ids])))
            fout_d.write("{}|{}\n".format(fid, ' '.join([str(x) for x in piece_dur])))


if __name__ == '__main__':
    fire.Fire(main)