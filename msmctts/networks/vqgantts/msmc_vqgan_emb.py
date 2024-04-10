from torch import nn
from torch.nn import functional as F

import numpy as np
import torch

from msmctts.utils.utils import get_mask_from_lengths
from msmctts.networks.acoustic_models.transformer import FFTBlocks
from msmctts.networks.hifigan import HifiGANGenerator
from .tdnn import ECAPA_TDNN
from .msmc_vqgan_speech import *


class AttrPredictor(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=5,
                dilation_rate=1,
                n_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.enc = ResStack(in_channels, kernel_size,
                            dilation_rate, n_layers)
        self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x, x_lengths):
        x = x.transpose(1, 2)
        x_mask = ~get_mask_from_lengths(x_lengths).unsqueeze(1)
        x = self.enc(x, x_mask)
        o = self.proj(x) * x_mask
        return x.transpose(1, 2), o.transpose(1, 2)


class MAMSEncoder(nn.Module):
    def __init__(self, in_channels,
                pitch_dim=1, energy_dim=1,
                downsample_scales=[1],
                max_seq_len=2400,
                n_layers=4,
                n_head=2,
                d_k=64,
                d_v=64,
                d_inner=1024,
                fft_conv1d_kernel=3,
                fft_conv1d_padding=1,
                dropout=0.2,
                attn_dropout=0.1,
                fused_layernorm=False):
        super(MAMSEncoder, self).__init__()
        self.downsample_scales = downsample_scales
        self.encoders = nn.ModuleList([
            FFTBlocks(max_seq_len=max_seq_len,
                      n_layers=n_layers,
                      n_head=n_head,
                      d_k=d_k,
                      d_v=d_v,
                      d_model=in_channels,
                      d_inner=d_inner,
                      fft_conv1d_kernel=fft_conv1d_kernel,
                      fft_conv1d_padding=fft_conv1d_padding,
                      dropout=dropout,
                      attn_dropout=attn_dropout,
                      fused_layernorm=fused_layernorm,
                      name='encoder_{}'.format(i))
        for i in range(len(downsample_scales))])

        self.use_pitch = False
        if pitch_dim + energy_dim > 0:
            self.use_pitch = True
            self.pitch_encoder = nn.Sequential(
                nn.Conv1d(pitch_dim + energy_dim, in_channels, 7, padding=3),
                nn.Tanh(),
                nn.Conv1d(in_channels, in_channels, 3, padding=1),
                nn.Tanh(),
                nn.Conv1d(in_channels, in_channels, 3, padding=1),
                nn.Tanh(),
                nn.Conv1d(in_channels, in_channels, 1),
            )

    def forward(self, emb, input_length, pitch=None, energy=None):
        if self.use_pitch:
            pitch_encoding = self.pitch_encoder(
                torch.cat((pitch, energy), dim=-1).transpose(1, 2)
            ).transpose(1, 2)

        outputs, content = [], None
        feat, feat_length = emb, input_length
        for encoder, scale in zip(self.encoders, self.downsample_scales):
            if scale > 1:
                feat = F.avg_pool1d(feat.transpose(1, 2),
                    kernel_size=scale, stride=scale, ceil_mode=True).transpose(1, 2)
                
                # Down-sample pitch
                if self.use_pitch:
                    pitch_encoding = F.avg_pool1d(pitch_encoding.transpose(1, 2),
                        kernel_size=scale, stride=scale, ceil_mode=True).transpose(1, 2)

                feat_length = torch.ceil(feat_length / scale).int()

            pos = torch.Tensor([i + 1 for i in range(feat_length.max())]).view(
                1, -1).repeat((feat.shape[0], 1)).long().to(feat.device)
            pos.masked_fill_(get_mask_from_lengths(feat_length), 0)
            feat, _ = encoder(feat, pos)
            
            if len(outputs) == 0:
                content = feat

            # Add pitch information
            if self.use_pitch:
                feat = feat + pitch_encoding
            
            outputs.append((feat, feat_length))

        return outputs, content


class MSMCVQGANEmb(nn.Module):
    """ Hierarchical Vector-Quantized Representation """

    def __init__(self, emb_dim, n_model_size,
                 pitch_dim=1, energy_dim=1,
                 encoder_config=None,
                 quantizer_config=None,
                 global_encoder_config=None,
                 frame_decoder_config=None,
                 decoder_config=None,
                 pred_mel=False,
                 mel_dim=None):
        super(MSMCVQGANEmb, self).__init__()
        
        # Build Encoder
        self.in_linear = nn.Linear(emb_dim, n_model_size)
        self.encoder = MAMSEncoder(n_model_size,
            pitch_dim=pitch_dim, energy_dim=energy_dim,
            **encoder_config)

        if global_encoder_config is not None:
            name = global_encoder_config._name
            if name == 'ECAPA_TDNN':
                self.global_encoder = ECAPA_TDNN(
                    in_channels=mel_dim,
                    embd_dim=n_model_size,
                    channels=n_model_size)
            else:
                raise ValueError("Wrong global encoder: {}".format(name))
        
        self.quantizer = MultiStageQuantizer(
            n_model_size, encoder_config['downsample_scales'][:: -1],
            **quantizer_config)
        
        # Decoder
        decoder_config['num_mels'] = n_model_size
        self.decoder = HifiGANGenerator(**decoder_config)

        # (Optional) Frame Decoder
        if frame_decoder_config is not None:
            self.frame_decoder = FFTBlocks(
                d_model=n_model_size,
                name='frame_decoder',
                **frame_decoder_config)
        
        # (Optional) Mel predictor
        if pred_mel:
            self.mel_predictor = nn.Linear(n_model_size, 
                mel_dim if mel_dim is not None else emb_dim)

    def forward(self, emb, emb_length,
                pitch=None, energy=None, 
                mel=None, ref=None,
                window='full'):
        output_dict = {}

        # Encode
        encoder_inputs = self.in_linear(emb)
        encoder_states, content_reps = self.encoder(
            encoder_inputs, emb_length, pitch, energy)
        encoder_outputs, encoder_lengths = zip(*encoder_states)
        output_dict['encoder_outputs'] = encoder_outputs[:: -1]
        output_dict['encoder_lengths'] = encoder_lengths[:: -1]
        output_dict['content_representations'] = content_reps

        # Quantize
        quantizer_states = self.quantizer(encoder_states)
        decoder_inputs = quantizer_states['residual_output']
        output_dict['encoder_indices'] = quantizer_states['quantizer_indices']
        output_dict['encoder_diffs'] = quantizer_states['quantizer_diffs']
        output_dict['decoder_diffs'] = quantizer_states['predictor_diffs']

        # Global encoder
        if hasattr(self, 'global_encoder'):
            ref = mel if ref is None else ref
            global_embeddings = self.global_encoder(ref).unsqueeze(1)
            decoder_inputs = decoder_inputs + global_embeddings

        # Frame Decoder
        if hasattr(self, 'frame_decoder'):
            pos = torch.Tensor([i + 1 for i in range(emb_length.max())]).view(
                1, -1).repeat((emb.shape[0], 1)).long().to(emb.device)
            pos.masked_fill_(get_mask_from_lengths(emb_length), 0)
            decoder_inputs, _ = self.frame_decoder(decoder_inputs, pos)

        # Mel predictor
        if hasattr(self, 'mel_predictor'):
            mel_outputs = self.mel_predictor(decoder_inputs)
            output_dict['mel_outputs'] = mel_outputs

        # Waveform Generation
        if window is not None:
            if isinstance(window, (list, tuple)):
                decoder_inputs = torch.stack([
                    decoder_inputs[i, s: e] for i, s, e in window],
                    dim=0)
            decoder_outputs = self.decoder(decoder_inputs.transpose(1, 2)).transpose(1, 2)
            output_dict['decoder_outputs'] = decoder_outputs

        return output_dict

    def analysis(self, emb, emb_length, pitch=None, energy=None):
        # Encode
        encoder_inputs = self.in_linear(emb)
        encoder_states, content_reps = self.encoder(
            encoder_inputs, emb_length, pitch, energy)

        # Quantize
        quantizer_states = self.quantizer(encoder_states)

        if self.training:
            encoder_outputs, encoder_lengths = zip(*encoder_states)
            return {
                'encoder_outputs': encoder_outputs[:: -1],
                'encoder_lengths': encoder_lengths[:: -1],
                'encoder_indices': quantizer_states['quantizer_indices'],
                'encoder_diffs': quantizer_states['quantizer_diffs'],
                'decoder_diffs': quantizer_states['predictor_diffs'],
                'quantizer_states': quantizer_states,
                'content_representations': content_reps,
            }
        return quantizer_states

    def synthesis(self, quantizer_outputs, quantizer_lengths, ref=None):
        # Quantize
        quantizer_states = quantizer_outputs
        if not isinstance(quantizer_outputs, dict):
            quantizer_states = self.quantizer(zip(quantizer_outputs, quantizer_lengths),
                                              from_encoder=False)
        decoder_inputs = quantizer_states['residual_output']

        # (Optional) Global encoder
        if hasattr(self, 'global_encoder'):
            assert ref is not None
            global_embeddings = self.global_encoder(ref).unsqueeze(1)
            decoder_inputs = decoder_inputs + global_embeddings


        # Frame Decoder
        if hasattr(self, 'frame_decoder'):
            decoder_length = quantizer_lengths[-1]
            pos = torch.Tensor([i + 1 for i in range(decoder_length.max())]).view(
                1, -1).repeat((decoder_inputs.shape[0], 1)).long().to(decoder_inputs.device)
            pos.masked_fill_(get_mask_from_lengths(decoder_length), 0)
            decoder_inputs, _ = self.frame_decoder(decoder_inputs, pos)

        # Decode
        decoder_outputs = self.decoder(decoder_inputs.transpose(1, 2)).transpose(1, 2)

        if self.training:
            output_dict = {'decoder_outputs': decoder_outputs}
            if hasattr(self, 'mel_predictor'):
                mel_outputs = self.mel_predictor(decoder_inputs)
                output_dict['mel_outputs'] = mel_outputs
            return output_dict

        return decoder_outputs


    def compute_embedding_loss(self, quantizer_outputs, quantizer_lengths, quantizer_states,
                               methods=['mse'], loss_weights=[1.0]):
        pred_states = [{
            'predictor_outputs': quantizer_outputs[i],
            'target_outputs': quantizer_states['quantizer_outputs'][i],
            'target_indices': quantizer_states['quantizer_indices'][i],
            'target_lengths': quantizer_lengths[i]
        } for i in range(len(quantizer_outputs))]

        return self.quantizer.compute_embedding_loss(pred_states, methods, loss_weights)


class KMeansQuantizer(nn.Module):
    def __init__(self, model_path):
        super(KMeansQuantizer, self).__init__()
        with open(model_path, 'rb') as fin:
            import pickle
            model = pickle.load(fin)
            codewords = torch.Tensor(model.cluster_centers_).T
            embedding_dims, embedding_sizes = codewords.shape
            self.codewords = codewords
        self.quantizer = nn.ModuleList([
            Quantize(embedding_dims, embedding_sizes)])
        

    def forward(self, encoder_states, from_encoder=True):
        # Init
        quant_states, pred_states = [], []
        residual_output = None
        
        # Quantize in stages
        for i, (embedding, length) in enumerate(encoder_states):
            quantizer_input = embedding
            # Quantization
            self.quantizer[i].embed = self.codewords.to(quantizer_input.device)
            quant, diffs, indices = self.quantizer[i](
                quantizer_input, length, update=False)
            # Save states
            quant_states.append((quant, diffs, indices))

        # Output
        quant_outputs, quant_diffs, quant_indices = zip(*quant_states)
        output_dict = {
            'residual_output': residual_output,
            'quantizer_outputs': quant_outputs,
            'quantizer_diffs': quant_diffs,
            'quantizer_indices': quant_indices,
            'quantizer_lengths': [x[1] for x in encoder_states]
        }

        # Calculate prediction loss
        predictor_diffs = None
        output_dict['predictor_diffs'] = predictor_diffs

        return output_dict


class KMeansVQGANEmb(nn.Module):

    def __init__(self, emb_dim, n_model_size,
                 quantizer_path,
                 global_encoder_config=None,
                 frame_decoder_config=None,
                 decoder_config=None,
                 pred_mel=False,
                 mel_dim=None):
        super(KMeansVQGANEmb, self).__init__()
        
        if global_encoder_config is not None:
            name = global_encoder_config._name
            if name == 'ECAPA_TDNN':
                self.global_encoder = ECAPA_TDNN(
                    in_channels=mel_dim,
                    embd_dim=n_model_size,
                    channels=n_model_size)
            else:
                raise ValueError("Wrong global encoder: {}".format(name))
        
        self.quantizer = KMeansQuantizer(quantizer_path)
    
        self.in_linear = nn.Linear(emb_dim, n_model_size)
        
        # Decoder
        decoder_config['num_mels'] = n_model_size
        self.decoder = HifiGANGenerator(**decoder_config)

        # (Optional) Frame Decoder
        if frame_decoder_config is not None:
            self.frame_decoder = FFTBlocks(
                d_model=n_model_size,
                name='frame_decoder',
                **frame_decoder_config)
        
        # (Optional) Mel predictor
        if pred_mel:
            self.mel_predictor = nn.Linear(n_model_size, 
                mel_dim if mel_dim is not None else emb_dim)

    def forward(self, emb, emb_length,
                pitch=None, energy=None, 
                mel=None, ref=None,
                window='full'):
        output_dict = {}

        # Encode
        quantizer_states = self.quantizer([(emb, emb_length)])
        decoder_inputs = quantizer_states['quantizer_outputs'][-1]
        output_dict['encoder_indices'] = quantizer_states['quantizer_indices']
        decoder_inputs = self.in_linear(decoder_inputs)


        # Global encoder
        if hasattr(self, 'global_encoder'):
            ref = mel if ref is None else ref
            global_embeddings = self.global_encoder(ref).unsqueeze(1)
            decoder_inputs = decoder_inputs + global_embeddings

        # Frame Decoder
        if hasattr(self, 'frame_decoder'):
            pos = torch.Tensor([i + 1 for i in range(emb_length.max())]).view(
                1, -1).repeat((emb.shape[0], 1)).long().to(emb.device)
            pos.masked_fill_(get_mask_from_lengths(emb_length), 0)
            decoder_inputs, _ = self.frame_decoder(decoder_inputs, pos)

        # Mel predictor
        if hasattr(self, 'mel_predictor'):
            mel_outputs = self.mel_predictor(decoder_inputs)
            output_dict['mel_outputs'] = mel_outputs

        # Waveform Generation
        if window is not None:
            if isinstance(window, (list, tuple)):
                decoder_inputs = torch.stack([
                    decoder_inputs[i, s: e] for i, s, e in window],
                    dim=0)
            decoder_outputs = self.decoder(decoder_inputs.transpose(1, 2)).transpose(1, 2)
            output_dict['decoder_outputs'] = decoder_outputs

        return output_dict

    def analysis(self, emb, emb_length):
        # Encode
        quantizer_states = self.quantizer([(emb, emb_length)])
        decoder_inputs = quantizer_states['quantizer_outputs'][-1]
        decoder_inputs = self.in_linear(decoder_inputs)

        if self.training:
            encoder_outputs, encoder_lengths = zip(*quantizer_states)
            return {
                'encoder_outputs': encoder_outputs[:: -1],
                'encoder_lengths': encoder_lengths[:: -1],
                'encoder_indices': quantizer_states['quantizer_indices'],
                'quantizer_states': quantizer_states,
            }
        return quantizer_states

    def synthesis(self, quantizer_outputs, quantizer_lengths, ref=None):
        # Encode
        quantizer_states = self.quantizer(zip(quantizer_outputs, quantizer_lengths))
        decoder_inputs = quantizer_states['quantizer_outputs'][-1]
        decoder_inputs = self.in_linear(decoder_inputs)

        # (Optional) Global encoder
        if hasattr(self, 'global_encoder'):
            assert ref is not None
            global_embeddings = self.global_encoder(ref).unsqueeze(1)
            decoder_inputs = decoder_inputs + global_embeddings


        # Frame Decoder
        if hasattr(self, 'frame_decoder'):
            decoder_length = quantizer_lengths[-1]
            pos = torch.Tensor([i + 1 for i in range(decoder_length.max())]).view(
                1, -1).repeat((decoder_inputs.shape[0], 1)).long().to(decoder_inputs.device)
            pos.masked_fill_(get_mask_from_lengths(decoder_length), 0)
            decoder_inputs, _ = self.frame_decoder(decoder_inputs, pos)

        # Decode
        decoder_outputs = self.decoder(decoder_inputs.transpose(1, 2)).transpose(1, 2)

        if self.training:
            output_dict = {'decoder_outputs': decoder_outputs}
            if hasattr(self, 'mel_predictor'):
                mel_outputs = self.mel_predictor(decoder_inputs)
                output_dict['mel_outputs'] = mel_outputs
            return output_dict

        return decoder_outputs


class EmbVC(nn.Module):

    def __init__(self, emb_dim, n_model_size,
                 pitch_dim=1, energy_dim=1,
                 encoder_config=None,
                 global_encoder_config=None,
                 frame_decoder_config=None,
                 decoder_config=None,
                 pred_mel=False,
                 mel_dim=None):
        super(EmbVC, self).__init__()
        
        # Build Encoder
        self.in_linear = nn.Linear(emb_dim, n_model_size)
        self.encoder = MAMSEncoder(n_model_size,
            pitch_dim=pitch_dim, energy_dim=energy_dim,
            **encoder_config)

        if global_encoder_config is not None:
            name = global_encoder_config._name
            if name == 'ECAPA_TDNN':
                self.global_encoder = ECAPA_TDNN(
                    in_channels=mel_dim,
                    embd_dim=n_model_size,
                    channels=n_model_size)
            else:
                raise ValueError("Wrong global encoder: {}".format(name))
        
        # Decoder
        decoder_config['num_mels'] = n_model_size
        self.decoder = HifiGANGenerator(**decoder_config)

        # (Optional) Frame Decoder
        if frame_decoder_config is not None:
            self.frame_decoder = FFTBlocks(
                d_model=n_model_size,
                name='frame_decoder',
                **frame_decoder_config)
        
        # (Optional) Mel predictor
        if pred_mel:
            self.mel_predictor = nn.Linear(n_model_size, 
                mel_dim if mel_dim is not None else emb_dim)

    def forward(self, emb, emb_length,
                pitch=None, energy=None, 
                mel=None, ref=None,
                window='full'):
        output_dict = {}

        # Encode
        encoder_inputs = self.in_linear(emb)
        encoder_states, content_reps = self.encoder(
            encoder_inputs, emb_length, pitch, energy)
        encoder_outputs, encoder_lengths = zip(*encoder_states)
        output_dict['encoder_outputs'] = encoder_outputs[:: -1]
        output_dict['encoder_lengths'] = encoder_lengths[:: -1]
        output_dict['content_representations'] = content_reps
        decoder_inputs = encoder_outputs[-1]

        # Global encoder
        if hasattr(self, 'global_encoder'):
            ref = mel if ref is None else ref
            global_embeddings = self.global_encoder(ref).unsqueeze(1)
            decoder_inputs = decoder_inputs + global_embeddings

        # Frame Decoder
        if hasattr(self, 'frame_decoder'):
            pos = torch.Tensor([i + 1 for i in range(emb_length.max())]).view(
                1, -1).repeat((emb.shape[0], 1)).long().to(emb.device)
            pos.masked_fill_(get_mask_from_lengths(emb_length), 0)
            decoder_inputs, _ = self.frame_decoder(decoder_inputs, pos)

        # Mel predictor
        if hasattr(self, 'mel_predictor'):
            mel_outputs = self.mel_predictor(decoder_inputs)
            output_dict['mel_outputs'] = mel_outputs

        # Waveform Generation
        if window is not None:
            if isinstance(window, (list, tuple)):
                decoder_inputs = torch.stack([
                    decoder_inputs[i, s: e] for i, s, e in window],
                    dim=0)
            decoder_outputs = self.decoder(decoder_inputs.transpose(1, 2)).transpose(1, 2)
            output_dict['decoder_outputs'] = decoder_outputs

        return output_dict

    def analysis(self, emb, emb_length, pitch=None, energy=None):
        # Encode
        encoder_inputs = self.in_linear(emb)
        encoder_states, content_reps = self.encoder(
            encoder_inputs, emb_length, pitch, energy)

        # Quantize
        quantizer_states = self.quantizer(encoder_states)

        if self.training:
            encoder_outputs, encoder_lengths = zip(*encoder_states)
            return {
                'encoder_outputs': encoder_outputs[:: -1],
                'encoder_lengths': encoder_lengths[:: -1],
                'encoder_indices': quantizer_states['quantizer_indices'],
                'encoder_diffs': quantizer_states['quantizer_diffs'],
                'decoder_diffs': quantizer_states['predictor_diffs'],
                'quantizer_states': quantizer_states,
                'content_representations': content_reps,
            }
        return quantizer_states

    def synthesis(self, quantizer_outputs, quantizer_lengths, ref=None):
        # Quantize
        quantizer_states = quantizer_outputs
        if not isinstance(quantizer_outputs, dict):
            quantizer_states = self.quantizer(zip(quantizer_outputs, quantizer_lengths),
                                              from_encoder=False)
        decoder_inputs = quantizer_states['residual_output']

        # (Optional) Global encoder
        if hasattr(self, 'global_encoder'):
            assert ref is not None
            global_embeddings = self.global_encoder(ref).unsqueeze(1)
            decoder_inputs = decoder_inputs + global_embeddings


        # Frame Decoder
        if hasattr(self, 'frame_decoder'):
            decoder_length = quantizer_lengths[-1]
            pos = torch.Tensor([i + 1 for i in range(decoder_length.max())]).view(
                1, -1).repeat((decoder_inputs.shape[0], 1)).long().to(decoder_inputs.device)
            pos.masked_fill_(get_mask_from_lengths(decoder_length), 0)
            decoder_inputs, _ = self.frame_decoder(decoder_inputs, pos)

        # Decode
        decoder_outputs = self.decoder(decoder_inputs.transpose(1, 2)).transpose(1, 2)

        if self.training:
            output_dict = {'decoder_outputs': decoder_outputs}
            if hasattr(self, 'mel_predictor'):
                mel_outputs = self.mel_predictor(decoder_inputs)
                output_dict['mel_outputs'] = mel_outputs
            return output_dict

        return decoder_outputs


    def compute_embedding_loss(self, quantizer_outputs, quantizer_lengths, quantizer_states,
                               methods=['mse'], loss_weights=[1.0]):
        pred_states = [{
            'predictor_outputs': quantizer_outputs[i],
            'target_outputs': quantizer_states['quantizer_outputs'][i],
            'target_indices': quantizer_states['quantizer_indices'][i],
            'target_lengths': quantizer_lengths[i]
        } for i in range(len(quantizer_outputs))]

        return self.quantizer.compute_embedding_loss(pred_states, methods, loss_weights)