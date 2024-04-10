from fairseq import checkpoint_utils
from scipy import interpolate
from tqdm import tqdm

import fire
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
import soundfile as sf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/mnt/public/usr/haohanguo/tools/models/hubert/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt"

print("loading model(s) from {}".format(model_path))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
print("loaded model(s) from {}".format(model_path))
print(f"normalize: {saved_cfg.task.normalize}")

model = models[0]
model = model.to(device)
model = model.half()
model.eval()


def interpolate_nearest(data, step_size=0.625):
    # Create indices
    indices = np.arange(0, data.shape[0])
    new_indices = np.arange(0, data.shape[0], step_size)
    
    # Interpolate
    assert indices.shape[0] == data.shape[0]
    f = interpolate.interp1d(indices, data.T, kind='nearest', fill_value="extrapolate")
    output = f(new_indices).T

    return output.astype(np.float32)


def postprocess(feats, normalize=False):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats


def _process_utterance(path, mel_dir=None, interpolate=True):
    # get output file path
    fid = os.path.split(path)[-1].split('.')[0]
    output_path = f'{mel_dir}/{fid}.npy'

    wav, sr = sf.read(path)
    frame_samples = int(sr * 0.02)
    wav = np.pad(wav, (0, (frame_samples - len(wav) % frame_samples)),
                 mode='constant')

    feat = torch.from_numpy(wav).float()
    feat = postprocess(feat, normalize=saved_cfg.task.normalize)
    feats = feat.view(1, -1)
    padding_mask = (
        torch.BoolTensor(feats.shape).fill_(False)
    )
    inputs = {
        "source": feats.half().to(device),
        "padding_mask": padding_mask.to(device),
    }
    logits = model.extract_features(**inputs)[0][0].cpu().numpy()
    logits = np.pad(logits, ((0, 1), (0, 0)), 'edge')

    # Interpolate
    if interpolate:
        logits = interpolate_nearest(logits)

    if mel_dir is not None:
        np.save(output_path, logits)

    return logits


def main(wav_dir, mel_dir, n_jobs=1):
    os.makedirs(mel_dir, exist_ok=True)

    with torch.no_grad():
        cache = []
        for filename in tqdm(os.listdir(wav_dir)):
            wav_path = os.path.join(wav_dir, filename)
            _process_utterance(wav_path, mel_dir)



if __name__ == '__main__':
  fire.Fire(main)
  print('Completed.')