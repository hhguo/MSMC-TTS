import fire
import os
import yaml



def main(list_file, file_out, phone_file=None, dur_file=None, mel_dir=None):
    with open(list_file) as fin:
        fids = [x.strip() for x in fin.readlines()]

    if phone_file is not None:
        with open(phone_file) as fin:
            lines = [x.strip() for x in fin.readlines()]
            phone_dict = {}
            for line in lines:
                fid, phone = line.split('|')
                phone_dict[fid] = phone
    
    if dur_file is not None:
        with open(dur_file) as fin:
            lines = [x.strip() for x in fin.readlines()]
            dur_dict = {}
            for line in lines:
                fid, dur = line.split('|')
                dur_dict[fid] = dur

    output_dict = {}
    for fid in fids:
        data_dict = {}
        if phone_file is not None:
            data_dict['text'] = phone_dict[fid]
        if dur_file is not None:
            data_dict['dur'] = dur_dict[fid]
        if mel_dir is not None:
            data_dict['mel'] = os.path.join(mel_dir, fid + '.npy')
        output_dict[fid] = data_dict

    with open(file_out, 'w') as fout:
        yaml.dump(output_dict, fout)


if __name__ == '__main__':
    fire.Fire(main)