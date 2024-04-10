from tqdm import tqdm
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
import re
import torch

from voicemaker.datasets import build_dataset
from voicemaker.tasks import build_task
from voicemaker.utils.utils import to_model, feature_normalize


def compute_codebook_complexity(all_indices):
    codebook = {}

    for indices in all_indices:
        for i in range(indices.shape[0]):
            code = indices[i]
            if code not in codebook:
                codebook[code] = 0
            codebook[code] += 1

    num_used = np.asarray([v for k, v in codebook.items()])
    probs = num_used / num_used.sum()
    complexity = -(probs * np.log2(probs)).sum()
    print(len(codebook.keys()), complexity)


def test(task, testset,
         output_dir=None,
         n_jobs=1):
    dataloader = DataLoader(testset, batch_size=n_jobs,
        num_workers=0, shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset),
        collate_fn=(testset.collate_fn
            if hasattr(testset, 'collate_fn') else None)
    )

    # Startup task
    if torch.cuda.is_available():
        task = task.cuda()
    task.eval()
    
    # VQ features
    all_indices = []
    all_diffs = []

    # Multi-process or Single-process
    for batch_i, features in tqdm(enumerate(dataloader)):
        # Get IDs if the batch is sorted in collate_fn
        test_ids = [testset.id_list[x] for x in features.pop('_id')]

        # Model inference
        features = to_model(features)
 
        mel = features.pop('mel')
        mel_length = features.pop('mel_length')
        quantizer_states = task.autoencoder.analysis(
            mel, mel_length.int())
        features['feat'] = quantizer_states['quantizer_outputs']
        features['feat_length'] = quantizer_states['quantizer_lengths']

        # Prediction
        output_dict = task.predictor(**features)
        embed_ind = output_dict['pseudo_labels']
        vq_diff = output_dict['vq_diff']
        pre_quants = output_dict['pre_quant']

        # Save output features
        for i, test_id in enumerate(test_ids):
            index = embed_ind[i]
            diff = vq_diff[i]
            pre_quant = pre_quants[i]
            if isinstance(index, torch.Tensor):
                index = index.detach().cpu().numpy()
                diff = diff.detach().cpu().numpy()
                pre_quant = pre_quant.detach().cpu().numpy()
            all_indices.append(index)
            all_diffs.append(diff)
            if output_dir is not None:
                np.save(os.path.join(output_dir, 'pre_quant', test_id + '.npy'), pre_quant)
                np.save(os.path.join(output_dir, 'indices', test_id + '.npy'), index)

    compute_codebook_complexity(all_indices)
    if output_dir is not None:
        embedding = task.predictor.msmcr_encoder.vq.embed.detach().cpu().numpy()
        np.save(os.path.join(output_dir, 'embedding.npy'), embedding.T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", required=True)
    parser.add_argument('-c', "--config", default=None)
    parser.add_argument('-t', "--test_config", default=None)
    parser.add_argument('-o', "--output_dir", default=None)
    parser.add_argument('-j', "--jobs", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'pre_quant'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'indices'), exist_ok=True)
    
    # Check arguments
    if args.model is None and args.config is None:
        parser.error('at least one argument shoule be given: -m/--model, -c/--config')

    # Load task from checkpoint file
    task = build_task(args.config,
                      mode='infer',
                      checkpoint=args.model)
    task.pre_infer()

    # Build Test Dataset
    dataset_config = task.config.testset \
        if hasattr(task.config, 'testset') else \
        task.config.dataset
    if args.test_config is not None:
        dataset_config['id_list'] = args.test_config
    dataset = build_dataset(dataset_config, training=False)

    # Inference
    with torch.no_grad():
        test(task, dataset, n_jobs=args.jobs, output_dir=args.output_dir)


if __name__ == '__main__':
    main()