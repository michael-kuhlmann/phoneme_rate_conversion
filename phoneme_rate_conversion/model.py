import copy
from typing import Union, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import paderbox as pb
from paderbox.utils.nested import nested_merge
import padertorch as pt
from padertorch.data.utils import pad_tensor
from padertorch.summary import tbx_utils
from padertorch.contrib.je.modules.conv import CNN1d, CNN2d
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor

from phoneme_rate_conversion.modules.speech_rate import LocalSpeechRate


class EstimateLocalSpeechRate(pt.Model):

    def __init__(
        self, feature_extractor: pt.Module, lsr_extractor: LocalSpeechRate,
        context_net: pt.Module,
        projection: Union[nn.Module, pt.Module, None] = None, rnn=None,
        label_key: str = 'phones', l2_norm=False,
    ):
        """
        Predict local speech rate from a speech signal. Showed to work good
        with mel-spectrogram + delta + delta-delta input features.

        Arguments:
            feature_extractor: Extract features from input.
            lsr_extractor: Computes local speech rate from labels which is used
                as training target.
            context_net: Extract hidden context features from input features.
            projection: Project hidden features to target.
            rnn: Optionally run an RNN over context features.
            l2_norm: If True, apply L2 regularization before projection.
            label_key: String with a single or multiple labels separated by a
                "+" which are used for local speech rate extraction.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.lsr_extractor = lsr_extractor
        self.context_net = context_net
        self.rnn = rnn
        if len(label_key.split('+')) >= 2:
            assert projection is not None, 'projection cannot be None'
            self.projection = nn.ModuleDict({
                k: copy.deepcopy(projection).apply(self._re_initialize_weights)
                for k in label_key.split('+')
            })
        else:
            self.projection = projection
        self.label_key = label_key
        self.l2_norm = l2_norm

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['feature_extractor'] = {
            'factory': NormalizedLogMelExtractor,
            'add_deltas': True,
            'add_delta_deltas': True,
        }
        config['lsr_extractor'] = {
            'factory': LocalSpeechRate,
        }

    # https://stackoverflow.com/a/49433937/16085876
    @staticmethod
    def _re_initialize_weights(module: nn.Module):
        if hasattr(module, 'weight'):
            nn.init.xavier_uniform_(module.weight)

    def snapshot(self, prediction, local_speech_rate, seq_len, prefix=None):
        if prefix is None:
            prefix = self.label_key
        prefix = prefix + '_'
        with torch.no_grad():
            snapshots = {}
            if self.create_snapshot:
                import matplotlib
                matplotlib.use('Agg')
                from paderbox.visualization import plot
                with pb.visualization.axes_context(2) as axes:
                    for i in range(min(2, local_speech_rate.shape[0])):
                        plot.line(
                            local_speech_rate[i, :seq_len[i]].cpu(),
                            label=f'Orig', ax=axes[i]
                        )
                        plot.line(
                            prediction[i, :seq_len[i]].cpu(),
                            label=f'Prediction', ax=axes[i]
                        )
                    lsr_fig = tbx_utils.figure()
                snapshots.update({f'{prefix}lsr_fig': lsr_fig})
        return snapshots

    def project(self, h, label_key, seq_len=None):
        if isinstance(self.projection, CNN1d):
            h = rearrange(h, 'b t f -> b f t')
            prediction, _ = self.projection(h, seq_len)
            prediction = prediction.squeeze(1)
        elif self.projection is not None:
            prediction = self.projection[label_key](h).squeeze(-1)
        else:
            assert h.shape[-1] == 1, h.shape
            prediction = h.squeeze(-1)
        return prediction

    def compute_loss(
        self, inputs, x, h, seq_len, seq_len_h
    ) -> Tuple[torch.Tensor, dict]:
        label_keys = self.label_key.split('+')
        summary = {
            'snapshots': {},
            'scalars': {},
            'buffers': {},
        }
        mse = torch.zeros(x.shape[0]).to(x.device)
        local_speech_rates = []
        for i, label_key in enumerate(label_keys):
            # If len(label_keys) == 1, predict single rate (e.g., phoneme or
            # syllable rate)
            prediction = self.project(h, label_key, seq_len=seq_len_h)

            segment_starts = inputs[f'{label_key}_start_frames']
            segment_stops = inputs[f'{label_key}_stop_frames']
            num_labels = np.array([len(s) for s in segment_starts])
            # pad segment_starts and segment_stops
            segment_starts = np.stack(
                [pad_tensor(np.asarray(s), max(num_labels), 0) for s in
                 segment_starts]
            )
            segment_stops = np.stack(
                [pad_tensor(np.asarray(s), max(num_labels), 0) for s in
                 segment_stops]
            )

            local_speech_rate, momentary_speed, seq_len = self.lsr_extractor(
                x, segment_starts, segment_stops, seq_len
            )
            local_speech_rate = local_speech_rate.squeeze(1)
            local_speech_rates.append(local_speech_rate)

            _mse = F.mse_loss(
                prediction[:, :], local_speech_rate, reduction='none'
            )
            _mse = Mean(axis=1)(_mse, seq_len)
            mse += _mse
            summary['scalars'][f'mse_{label_key}'] = _mse.mean().detach()
            snapshots = self.snapshot(
                prediction, local_speech_rate, seq_len, prefix=label_key
            )
            summary['snapshots'].update(snapshots)
            with torch.no_grad():
                num_predicted_labels = self.lsr_extractor.get_num_segments(
                    prediction, seq_len_h)
                rel_diff = np.abs(
                    num_predicted_labels - num_labels) / num_labels
                summary['buffers'][f'rel_diff_{label_key}'] = rel_diff

        return mse, summary

    def encode(self, x, seq_len):
        if isinstance(self.context_net, CNN2d):
            c, seq_len_c = self.context_net(x, seq_len)
            if c.ndim == 4:
                c = rearrange(c, 'b c f t -> b (c f) t')
        else:
            c = rearrange(x, 'b c f t -> b (c f) t')
            c, seq_len_c = self.context_net(c, seq_len)

        c = rearrange(c, 'b f t -> b t f')
        if self.rnn is not None:
            h, _ = self.rnn(c)
        else:
            h = c
        return h, seq_len_c

    def forward(self, inputs):
        x = inputs['stft']
        seq_len = inputs['seq_len']

        x, seq_len = self.feature_extractor(x, seq_len)

        if self.feature_extractor.time_warping is not None and self.training:
            factors = np.asarray(seq_len) / np.asarray(inputs['seq_len'])
            for label_key in self.label_key.split('+'):
                start_frames = []
                stop_frames = []
                for factor, _start_frames, _stop_frames in zip(
                    factors, inputs[f'{label_key}_start_frames'],
                    inputs[f'{label_key}_stop_frames']
                ):
                    _start_frames = np.minimum(
                        np.ceil(np.asarray(_start_frames) * factor), x.shape[-1]
                    ).astype(np.int32)
                    _stop_frames = np.minimum(
                        np.ceil(np.asarray(_stop_frames) * factor), x.shape[-1]
                    ).astype(np.int32)
                    start_frames.append(_start_frames)
                    stop_frames.append(_stop_frames)
                inputs[f'{label_key}_start_frames'] = start_frames
                inputs[f'{label_key}_stop_frames'] = stop_frames

        h, seq_len_c = self.encode(x, seq_len)
        mse, _summary = self.compute_loss(inputs, x, h, seq_len, seq_len_c)
        summary = {
            'losses': {
                'mse': mse.mean(),
            },
            'histograms': {
                'mse_': mse.flatten(),
            },
        }
        if self.l2_norm:
            l2_penalty = Mean(axis=-1)(
                torch.linalg.norm(h, ord=2, dim=-1), seq_len_c
            )
            summary['losses']['l2_penalty'] = l2_penalty.mean()
        summary = nested_merge(summary, _summary)

        return summary

    def review(self, inputs, outputs):
        return outputs

    def modify_summary(self, summary):
        if len(summary['snapshots']) > 0:
            for key in summary['snapshots'].keys():
                lsr_fig = summary['snapshots'][key]
                summary['figures'].update({key: lsr_fig})
            summary['snapshots'].clear()
        if len(summary['buffers']) > 0:
            for label_key in self.label_key.split('+'):
                rel_diff = np.concatenate(summary['buffers'].pop(
                    f'rel_diff_{label_key}'))
                summary['scalars'][f'rel_diff_{label_key}'] = np.mean(rel_diff)
        return super().modify_summary(summary)

    def predict(self, x, seq_len=None, label_key=None):
        """

        Args:
            x: STFT signal
            seq_len:
            label_key:

        Returns: torch.Tensor
            Predicted speech rate
        """
        with torch.no_grad():
            x, seq_len = self.feature_extractor(x, seq_len)
            h, seq_len_c = self.encode(x, seq_len)
            label_keys = self.label_key.split('+')
            if label_key is None:
                label_key = label_keys[0]
            prediction = self.project(h, label_key, seq_len=seq_len_c)

        return prediction
