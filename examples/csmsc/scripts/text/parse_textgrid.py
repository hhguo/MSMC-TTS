from tqdm import tqdm

import fire
import librosa
import os

from symbols import symbols


symbol_to_id = {s: i for i, s in enumerate(symbols)}


def parse_textgrid(file_textgrid, file_wav):
    with open(file_textgrid) as fin:
        lines = [x.strip() for x in fin.readlines()]
    
    content = lines[12:]
    start = content[:: 3]
    end = content[1:: 3]
    phones = [s.strip('"') for s in content[2:: 3]]

    phone_inds, dur = [], []
    for i in range(len(phones)):
        phone, tone, er = phones[i], '0', '0'
        if phone[: 2] != 'sp' and phone[-1:] in '0123456789':
            tone = phone[-1]
            phone = phone[: -1]
        if phone != 'er' and phone[-1] == 'r' and phone[: -1] in symbol_to_id:
            er = '1'
            phone = phone[: -1]

        phone_inds.append(f'{str(symbol_to_id[phone])}_{tone}_{er}')
        dur.append(float(end[i]) - float(start[i]))


    dur = [d * 80 for d in dur]
    rest = 0
    for i in range(len(dur)):
        dur[i] += rest
        rest = dur[i] - round(dur[i])
        dur[i] = str(round(dur[i]))

    wav_dur = librosa.get_duration(filename=file_wav)
    if abs(wav_dur - float(end[-1])) > 0.1:
        print(file_textgrid, wav_dur, end)
    
    return ' '.join(phone_inds), ' '.join(dur)
        

def main(dir_textgird, dir_wav, file_text, file_dur):
    output_dict = {}

    for filename in tqdm(os.listdir(dir_textgird)):
        file_id = filename.split('.')[0]
        file_textgrid = os.path.join(dir_textgird, filename)
        file_wav = os.path.join(dir_wav, file_id + '.wav')
        
        text, duration = parse_textgrid(file_textgrid, file_wav)
        
        output_dict[file_id] = {
            'text': text,
            'dur': duration
        }
    
    with open(file_text, 'w') as fout:
        for fid in sorted(output_dict.keys()):
            text = output_dict[fid]['text']
            fout.write(f"{fid}|{text}\n")
    
    with open(file_dur, 'w') as fout:
        for fid in sorted(output_dict.keys()):
            dur = output_dict[fid]['dur']
            fout.write(f"{fid}|{dur}\n")


if __name__ == '__main__':
    fire.Fire(main)