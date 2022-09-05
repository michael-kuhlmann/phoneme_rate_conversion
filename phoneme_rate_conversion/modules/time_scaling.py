import dataclasses
from typing import Union, Tuple, List
import math

import numpy as np
from scipy.signal import get_window
import torch
from paderbox.utils.misc import interleave
import padertorch as pt


class LinearIndicesSampler:

    def __init__(self, sampling_fn):
        self.sampling_fn = sampling_fn

    def __call__(self, seq_len, factor=None):
        if factor is None:
            resampling_factor = self.sampling_fn(len(seq_len))
        else:
            resampling_factor = factor
        return self.resample(seq_len, resampling_factor)

    def resample(self, seq_len, factor):
        new_len = np.ceil(factor * seq_len).astype(np.int)
        time_indices = []
        for j, _len in enumerate(new_len):
            new_indices = np.linspace(0, seq_len[j] - 1, _len, endpoint=False)
            time_indices.append(torch.from_numpy(new_indices))
        time_indices = torch.nn.utils.rnn.pad_sequence(time_indices).numpy().T
        return time_indices, new_len


@dataclasses.dataclass
class WSOLA:
    sample_rate: int
    window_length_in_ms: float = 0.03
    max_tol: float = 0.005

    def __post_init__(self):
        self.window_length = int(self.window_length_in_ms * self.sample_rate)
        self.max_tol = int(self.max_tol * self.sample_rate)

    def _get_indices(self, segment_centers, segment_length, max_tol):
        indices = np.array([
            np.arange(
                segment_centers - segment_length + tol,
                segment_centers + segment_length + tol
            ) for tol in interleave(-np.arange(max_tol), range(1, max_tol))
        ])  # search first for deltas that are close to segment_center
        return indices

    def _stretch(
        self, x, segment_length, segment_left, segment_centers,
        window_length, max_tol
    ):
        if isinstance(
            segment_left, (np.ndarray, torch.Tensor)
        ) and segment_left.ndim == 1:
            _window_length = min(
                x.shape[0] - max(segment_left).item(), window_length
            )
            if not isinstance(segment_left, torch.Tensor):
                _segment_left = torch.tensor(segment_left)
            else:
                _segment_left = segment_left
            x_cont = x[
                (
                    _segment_left[:, None].cpu()
                    +torch.arange(_window_length)[None]
                ).T,
                np.arange(len(segment_left))
            ]
        else:
            x_cont = x[segment_left:segment_left+window_length]
        _max_tol = np.minimum(
            x.shape[0] - (segment_centers+segment_length), max_tol
        )
        indices = self._get_indices(segment_centers, segment_length, _max_tol)
        try:
            x_cand = x[indices]
            if x_cand.shape[0] == 0 or x_cont.shape[0] != self.window_length:
                return None, None
        except IndexError:
            return None, None
        if isinstance(x_cand, torch.Tensor):
            scale = torch.sqrt((x_cand**2).sum(axis=1)) + 1e-6
        else:
            scale = np.sqrt((x_cand ** 2).sum(axis=1)) + 1e-6
        corr = (
            (x_cont[None] * x_cand).sum(axis=1) / scale
        )
        arg = corr.argmax(axis=0)
        if arg.ndim == 1:
            x_best = x_cand[arg, ..., np.arange(len(arg))]
            if isinstance(x_best, torch.Tensor):
                x_best = x_best.transpose(0, -1)
            else:
                x_best = np.moveaxis(x_best, 0, 1)
        else:
            x_best = x_cand[arg]
        return x_best, arg

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor], factors: float,
        sequence_lengths: List[int] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], List[int]]:
        """

        Args:
            x: Single- or multi-channel audio signal. Shape is (N,) or (N, C)
            factors: Single scale factor. A factor > 1.0 compresses the signal,
                otherwise stretch
            sequence_lengths: Length of each audio channel

        Returns:
            Time-scale modified output signal and post-scale sequence lengths
        """
        window_length = self.window_length  # N
        max_tol = self.max_tol
        segment_length = window_length // 2  # L
        w = get_window('hann', window_length).reshape(
            (window_length, *[1]*(x.ndim-1))
        )
        if isinstance(x, torch.Tensor):
            w = torch.from_numpy(w).to(x.device)
        y = [w[segment_length:] * x[:segment_length]]
        segment_left = segment_length
        k = 1
        while all(pt.utils.to_list(segment_left + window_length)) < x.shape[0]:
            segment_centers = math.floor(k * segment_length * factors)
            x_best, arg = self._stretch(
                x, segment_length, segment_left, segment_centers,
                window_length, max_tol
            )
            if x_best is None:
                break
            x_best = w * x_best
            y[-1] = y[-1] + x_best[:segment_length]
            y.append(x_best[segment_length:])
            # update parameters for next segment
            delta_k = ((arg + 1) // 2) * (-1) ** ((arg+1) % 2)
            segment_left = segment_centers + delta_k
            if (
                isinstance(segment_left, (np.ndarray, torch.Tensor))
                and segment_left.ndim == 0
            ):
                segment_left = segment_left.item()
            k += 1

        if isinstance(x, torch.Tensor):
            y = torch.cat(y).float()
        else:
            y = np.concatenate(y)
        if sequence_lengths is not None:
            sequence_lengths = np.floor(
                np.asarray(sequence_lengths) * (y.shape[0] / x.shape[0])
            ).astype(np.int32).tolist()
        return y, sequence_lengths
