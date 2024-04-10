from time import time

import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from voicemaker.utils.plot import plot_matrix
from voicemaker.utils.utils import get_mask_from_lengths
from .base_trainer import BaseTrainer
from .msmctts_trainer import VQGANTrainer


class EmbVQGANTrainer(VQGANTrainer):
    def __init__(self, *args,
                 sample_batch_size=-1,
                 frame_loss_supervised_step=0,
                 lambda_frame=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_batch_size = sample_batch_size
        self.lambda_frame = lambda_frame
        self.frame_loss_supervised_step = frame_loss_supervised_step
        self.frameshift = self.config.dataset.frameshift[self.config.dataset.feature.index('mel')]
        self.frame_lengths = -1 if self.sample_lengths == -1 else \
            self.sample_lengths // self.frameshift

    def train_step(self, batch, iteration):
        losses = {}
        emb, emb_length = batch['emb'], batch['emb_length']
        wav, wav_length = batch['wav'], batch['wav_length']
        mel, mel_length = batch['mel'], emb_length
        pitch = batch['pitch'] if 'pitch' in batch else None
        energy = batch['energy'] if 'energy' in batch else None

        # Window-based Waveform Generation
        if iteration > self.frame_loss_supervised_step:
            # Create windows
            # 1. Sample N sequences from the batch
            seq_indices = range(wav_length.shape[0])
            if self.sample_batch_size > 0:
                seq_indices = list(range(wav_length.shape[0]))
                random.shuffle(seq_indices)
                seq_indices = seq_indices[: self.sample_batch_size]
                seq_indices.sort()
            
            # 2. Sample windows from each sampled sequence
            windows, window_wavs = [], []
            for i in seq_indices:
                start = random.randrange(max(1, mel_length[i] - self.frame_lengths))
                end = start + self.frame_lengths
                windows.append((i, start, end))
                start, end = start * self.frameshift, end * self.frameshift
                window_wavs.append(wav.squeeze()[i, start: end])
            target = torch.stack(window_wavs, dim=0).squeeze()
        else:
            windows = None

        # Forward
        output = self.model.autoencoder(emb, emb_length,
                                        pitch, energy,
                                        mel=mel,
                                        window=windows)

        g_loss = 0
        # VQ loss
        if 'encoder_indices' in output:
            vq_loss = self.vq_criterion(output)
            losses.update(vq_loss)
            g_loss = g_loss + vq_loss['vq_loss']

        # Frame Loss
        if 'mel_outputs' in output:
            pred_mel = output['mel_outputs']
            mel_loss = F.mse_loss(mel, pred_mel, reduction='none')
            mel_mask = get_mask_from_lengths(mel_length)
            mel_loss.masked_fill_(mel_mask.unsqueeze(-1), 0)
            mel_loss = mel_loss.sum() / sum(mel_length) / mel_loss.shape[2]
            losses['frame_loss'] = mel_loss.item()
            g_loss = g_loss + self.lambda_frame * mel_loss

        # MR-STFT Loss
        if 'decoder_outputs' in output:
            predict = output['decoder_outputs'].squeeze()
            stft_loss = self.stft_criterion(predict, target)
            if isinstance(stft_loss, dict):
                stft_loss_sum = 0
                for name, term in stft_loss.items():
                    stft_loss_sum = stft_loss_sum + term
                    losses[name] = term.item()
                stft_loss = stft_loss_sum
            losses['stft_loss'] = stft_loss.item()
            g_loss = g_loss + self.lambda_stft * stft_loss

        # Adversarial loss for prosody estimator
        if hasattr(self.model, 'prosody_estimator'):
            content = output['content_representations']
            
            # Train Discriminator
            pred_prosody = self.model.prosody_estimator(content.detach())
            prosody_loss = F.mse_loss(torch.cat((pitch, energy), dim=-1),
                pred_prosody, reduction='none')
            prosody_loss.masked_fill_(mel_mask.unsqueeze(-1), 0)
            prosody_loss = prosody_loss.sum() / sum(mel_length) / prosody_loss.shape[2]
            losses['d_prosody_loss'] = prosody_loss
            prosody_loss *= 0.01

            self.optimizer.zero_grad(['prosody_estimator'])
            prosody_loss.backward()
            self.optimizer.step(['prosody_estimator'])

            # Train Generator
            pred_prosody = self.model.prosody_estimator(content)
            prosody_loss = F.mse_loss(torch.cat((pitch, energy), dim=-1),
                pred_prosody, reduction='none')
            prosody_loss.masked_fill_(mel_mask.unsqueeze(-1), 0)
            prosody_loss = prosody_loss.sum() / sum(mel_length) / prosody_loss.shape[2]
            losses['g_prosody_loss'] = prosody_loss
            g_loss = g_loss + 0.01 * -1 * prosody_loss

        # Adversarial Loss
        if iteration > self.stft_loss_supervised_step:
            # Train discriminator
            fake_scores, fake_feats = self.model.discriminator(predict.detach())
            real_scores, real_feats = self.model.discriminator(target)

            d_loss_fake_list, d_loss_real_list = [], []
            for (real_score, fake_score) in zip(real_scores, fake_scores):
                d_loss_real_list.append(self.criterion(real_score, torch.ones_like(real_score)))
                d_loss_fake_list.append(self.criterion(fake_score, torch.zeros_like(fake_score)))
            d_loss_real = sum(d_loss_real_list)
            d_loss_fake = sum(d_loss_fake_list)
            d_loss = d_loss_real + d_loss_fake

            losses['d_loss_real'] = d_loss_real.item()
            losses['d_loss_fake'] = d_loss_fake.item()
            losses['d_loss'] = d_loss.item()

            self.optimizer.zero_grad(['discriminator'])
            d_loss.backward()
            self.optimizer.step(['discriminator'])
        
            # Train Generator
            fake_scores, fake_feats = self.model.discriminator(predict)
            real_scores, real_feats = self.model.discriminator(target)
            
            adv_loss_list = []
            feat_loss_list = []
            for fake_score in fake_scores:
                adv_loss_list.append(self.criterion(fake_score, torch.ones_like(fake_score)))
            for i in range(len(fake_feats)):
                for j in range(len(fake_feats[i])):
                    feat_loss_list.append(self.l1_loss(fake_feats[i][j], real_feats[i][j]))
            adv_loss = sum(adv_loss_list) # / len(adv_loss_list)
            feat_loss = sum(feat_loss_list) # / len(feat_loss_list)
            adv_loss = adv_loss + feat_loss * (
                self.lambda_fm if self.lambda_fm != 'auto' else
                (g_loss / feat_loss).detach())
            g_loss = g_loss + adv_loss

            losses['fm_loss'] = feat_loss.item()
            losses['adv_loss'] = adv_loss.item()
            losses['g_loss'] = g_loss.item()

        self.optimizer.zero_grad(['autoencoder'])
        g_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.autoencoder.parameters(), self.grad_clip_thresh)
        self.optimizer.step(['autoencoder'])

        return {'loss': losses}


from voicemaker.tasks import load_model
from .fastspeech_trainer import FSLoss
class NASynEmbFSTrainer(BaseTrainer):
    def __init__(self, config, model, num_gpus=1, rank=0,
                 grad_clip_thresh=1.0,
                 eval_inteval_iters=1000,
                 training_methods=['mse'],
                 loss_weights=[1.0],
                 lambda_dur=1.0,
                 ):
        super().__init__(config, model, num_gpus, rank)
        self.training_methods = training_methods
        self.loss_weights = loss_weights
        self.grad_clip_thresh = grad_clip_thresh
        self.eval_inteval_iters = eval_inteval_iters
        self.fs_loss = FSLoss(lambda_dur)

    def train_step(self, batch, iteration):
        # Init loss
        losses = {'total_loss': 0}

        # Extract embedding features
        if not hasattr(self, 'autoencoder'):
            self.build_autoencoder()

        self.autoencoder.eval()
        self.autoencoder.require_grad = False
        with torch.no_grad():
            quantizer_states = self.autoencoder.analysis(
                batch.pop('emb'),
                batch.pop('emb_length').int(),
                pitch=batch.pop('pitch') if 'pitch' in batch else None,
                energy=batch.pop('energy') if 'energy' in batch else None,
            )
            batch['feat'] = quantizer_states['quantizer_outputs']
            batch['feat_length'] = quantizer_states['quantizer_lengths']

        # Model forward
        output = self.model.predictor(**batch)
        
        # Embedding loss
        pred = output['feat']
        length = output['feat_length']
        embedding_loss = self.autoencoder.compute_embedding_loss(
            pred, length, quantizer_states,
            methods=self.training_methods,
            loss_weights=self.loss_weights)
        losses['total_loss'] += embedding_loss.pop('total_loss')
        losses.update(embedding_loss)

        # FastSpeech loss
        fs_loss = self.fs_loss(output, batch)
        losses['total_loss'] += fs_loss.pop('total_loss')
        losses.update(fs_loss)

        # Update parameters
        self.optimizer.zero_grad(['predictor'])
        losses['total_loss'].backward()
        if self.grad_clip_thresh is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.predictor.parameters(),
                self.grad_clip_thresh)
            losses['grad_norm'] = grad_norm
        self.optimizer.step(['predictor'])

        # Write logs
        log = {'loss': losses}
        
        return log

    def build_autoencoder(self):
        checkpoint = self.config.task.autoencoder._checkpoint
        config = self.config.task.autoencoder._config \
            if hasattr(self.config.task.autoencoder, '_config') \
            else None
        self.autoencoder = load_model('autoencoder', checkpoint, config)
        if torch.cuda.is_available():
            self.autoencoder = self.autoencoder.cuda()
