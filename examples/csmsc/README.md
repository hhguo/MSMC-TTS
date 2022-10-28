# CSMSC

A Chinese single-speaker TTS dataset, with 12-hour recordings, corresponding transcripts, phonemes, durations, and prosody labels.

## Data
Data Link: https://www.data-baker.com/data/index/TNtts

## Pre-processing
Please follow examples/csmsc/scripts/process_dataset.sh to process the dataset to obtain these features:

|  Feature   | format  | Description |
|  ----  | ----  | ---- |
| waveform  | .wav | 24k sample rate, single channel |
| Mel spectrogram | .npy  | 80-dim, 12.5ms frameshift, normed value [-4, 4] |
| phoneme | .txt | Phoneme indices (phoneme + tone + 1-dim rhotic label) |
| duration | .txt | The number of frames for each phoneme |

The file list for training is an ID list with the format:
```text
000001
000002
000003
000004
```

The file list for test is a .yaml file with the format:
```yaml
000001:
    text: 
000002:
    text:
```

## Pre-trained Models
You may download pre-trained models:

| Model | Link |
| ----  | ---- |
| Standard MSMC-TTS | MSMC-VQ-GAN:  |