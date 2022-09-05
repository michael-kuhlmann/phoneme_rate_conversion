from typing import Union, List, Tuple

from scipy import signal
import scipy.integrate as integrate
import numpy as np
import torch
import torch.nn.functional as F

import padertorch as pt
from padertorch.ops.sequence.mask import compute_mask
from padertorch.contrib.je.modules.conv_utils import compute_pad_size


class LocalSpeechRate(pt.Module):

    def __init__(
        self, rate: Union[int, float], window_size: int, shift: int = 1,
        pad_type: str = 'both',
    ):
        """
        Extract local speech rate from acoustic units (phonemes, syllables,
        words, ...) according to [1].

        [1]: Pfitzinger, Hartmut R. "Reducing Segmental Duration Variation by
        Local Speech Rate Normalization of Large Spoken Language Resources."
        In LREC. 2002.

        Args:
            rate: Input rate of the signal.
            window_size: Total number of context samples or frames.
            shift: Shift between successive windows. Defaults to 1.
            pad_type: Controls convolution behavior.
                    None: No padding.
                    front: Causal convolution.
                    end: Append zeros before convolution.
                    both: Add window_length / 2 zeros to both sides of the
                        signal.
                When padding is None, the output length is
                `(T - self.window_size + 1) / self.shift`, where `T` is the
                input length of the signal. Otherwise, the output length is
                `T / self.shift`.
        """
        super().__init__()
        self.sr = rate
        window_size = int(window_size / shift)
        if window_size % 2 == 0:
            # Ensure odd number of samples in window
            window_size += 1
        self.window_size = window_size
        self.shift = shift
        w = torch.from_numpy(
            signal.windows.hann(self.window_size, sym=False)
        )[None, None].float()
        self.register_buffer('window', w)
        self.pad_type = pad_type

    def get_num_segments(
        self, lsr: Union[torch.Tensor, np.ndarray],
        sequence_lengths: Union[List[int], None] = None,
    ) -> np.ndarray:
        """
        Integrate over area-under-curve to get an estimation for the number of
        segments in the utterance.

        Args:
            lsr: Predicted local speech rate (`v_t`) from forward step. Shape
                (B, T) or (T,)
            sequence_lengths: Actual lengths of each batch element without
                padding.

        >>> x = torch.zeros((1, 32_800))
        >>> segment_starts = np.array([0.25, 0.83, 1.14])[None] * 16_000
        >>> segment_stops = np.array([0.8, 1.14, 1.78])[None] * 16_000
        >>> lsr_extractor = LocalSpeechRate(16_000, int(0.625 * 16_000), shift=int(0.0125 * 16_000))
        >>> v_t, v_m_t, _ = lsr_extractor(x, segment_starts, segment_stops)
        >>> lsr_extractor.get_num_segments(v_t.squeeze(1))
        array([3])
        """
        if lsr.ndim > 2:
            raise TypeError(
                'Input has too many dimensions! Expected at most 2 but got '
                f'{lsr.ndim} (input shape is {lsr.shape})'
            )
        lsr = lsr / self.sr * self.shift
        mask = compute_mask(lsr, sequence_lengths, sequence_axis=lsr.ndim - 1)
        lsr = lsr * mask
        if isinstance(lsr, torch.Tensor):
            lsr = pt.utils.to_numpy(lsr, detach=True)
        pred_segments = np.round(
            integrate.simpson(lsr, axis=lsr.ndim - 1)).astype('int')
        return pred_segments

    def forward(
        self,
        x: torch.Tensor,
        segment_starts: Union[np.ndarray, torch.Tensor],
        segment_stops: Union[np.ndarray, torch.Tensor],
        sequence_lengths: Union[List[int], np.ndarray, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Compute speech rate from alignment information.

        Args:
            x: Input signal of shape (B, *, T).
            segment_starts: Start timings of non-silence speech units.
                Shape: (B, N).
            segment_stops: Stop timings of non-silence speech units.
                Shape: (B, N).
            sequence_lengths: List with number of speech units for each batch
                element.

        Returns:
            Local speech rate, instantaneous speech rate and length of each
            speech signal without padding.

        >>> x = torch.zeros((1, 32_800))
        >>> segment_starts = np.array([0.25, 0.83, 1.14]) * 16_000
        >>> segment_stops = np.array([0.8, 1.14, 1.78]) * 16_000
        >>> v_t, v_m_t, _ = LocalSpeechRate(16_000, int(0.625 * 16_000), shift=int(0.0125 * 16_000))(x, segment_starts[None], segment_stops[None])
        >>> v_t.shape
        torch.Size([1, 1, 164])
        >>> stft = pb.transform.module_stft.STFT(shift=200, size=1024, window_length=800, fading='half')
        >>> x = np.moveaxis(stft(x), 1, 2)
        >>> segment_starts = np.array([stft.sample_index_to_frame_index(s) for s in segment_starts])
        >>> segment_stops = np.array([stft.sample_index_to_frame_index(s) for s in segment_stops])
        >>> lsr_extractor = LocalSpeechRate(80, int(0.625*80), shift=1)
        >>> v_t, v_m_t, _ = lsr_extractor(np.abs(x), segment_starts[None], segment_stops[None])
        >>> v_t.shape
        torch.Size([1, 1, 164])
        """
        device = self.window.device
        if isinstance(segment_starts, list) or isinstance(segment_stops, list):
            raise TypeError(
                "Got type 'list' for segment_starts or segment_stops.\n"
                "Expected np.ndarray or torch.Tensor.\n"
                "Convert segment_starts and segment_stops to ndarray and apply "
                "zero-padding if necessary. You can provide the number of "
                "non-padded speech units through the 'sequence_lengths' "
                "argument."
            )
        if isinstance(segment_starts, np.ndarray):
            segment_starts = torch.from_numpy(segment_starts).to(device)
        if isinstance(segment_stops, np.ndarray):
            segment_stops = torch.from_numpy(segment_stops).to(device)
        durations = (segment_stops - segment_starts).to(device).float()
        # Instantaneous speech rate
        v_m = torch.where(
            durations > 0, 1 / durations,
            torch.tensor(0, dtype=torch.float32).to(durations)
        )
        if sequence_lengths is None:
            sequence_lengths = np.full(
                segment_starts.shape[0], fill_value=segment_starts.shape[1]
            )
        range_t = torch.arange(0, x.shape[-1], step=self.shift).unsqueeze(0)\
            .expand((v_m.shape[0], -1)).float().to(device)
        v_m_t = torch.zeros_like(range_t)
        seq_len = []
        for i in range(v_m.shape[0]):
            starts_i = segment_starts[i, :sequence_lengths[i]]
            stops_i = segment_stops[i, :sequence_lengths[i]]
            ind_t, ind_seg = torch.where(
                (range_t[i, :, None] >= starts_i)
                & (range_t[i, :, None] < stops_i)
            )
            v_m_t[i, ind_t] = v_m[i, ind_seg]
            seq_len.append(len(ind_seg))
        v_m_t = v_m_t.unsqueeze(1)  # (B, 1, T)
        with torch.no_grad():
            pad_size = compute_pad_size(self.window_size, 1, 1, self.pad_type)
            _v_m_t = F.pad(v_m_t, pad_size, 'constant')
            # Local speech rate
            v_t = F.conv1d(
                _v_m_t.to(device), self.window, padding=0
            )  # (B, 1, T)
            # Convert to number of units in seconds and re-scale start and end
            # in case of same padding
            ones = torch.ones_like(v_m_t).to(device)
            ones = F.pad(ones, pad_size, 'constant')
            norm = F.conv1d(ones, self.window, padding=0)
            v_t = v_t / norm * self.sr
        seq_len = np.array(seq_len)
        return v_t, v_m_t * self.sr, seq_len
