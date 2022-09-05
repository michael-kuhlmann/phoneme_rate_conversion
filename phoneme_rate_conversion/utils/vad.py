import dataclasses
from math import ceil
from collections import deque

import numpy as np
import webrtcvad
from paderbox.array.interval import ArrayInterval
import padertorch as pt


@dataclasses.dataclass
class WebRTCVAD(pt.Configurable):
    sample_rate: int
    mode: int = 2
    threshold: float = 0.95
    frame_duration_in_ms: float = 30.0
    padding_duration_in_ms: float = 300.0
    axis: int = -1
    reset: bool = True

    def __post_init__(self):
        self.vad = webrtcvad.Vad(self.mode)
        self.frame_duration = int(
            self.sample_rate * self.frame_duration_in_ms / 1000)
        self.padding_duration = int(
            self.sample_rate * self.padding_duration_in_ms / 1000)

    def __call__(self, signal: np.ndarray) -> ArrayInterval:
        shape = signal.shape
        if signal.ndim <= 2:
            if signal.ndim == 2 and shape[self.axis % signal.ndim - 1] != 1:
                raise ValueError(
                    'Expected mono signal with at least two dimensions but '
                    f'got signal with shape {signal.shape}'
                )
        else:
            raise ValueError(
                'Expected mono signal with at least two dimensions but '
                f'got signal with shape {signal.shape}'
            )
        pad_width = np.zeros((signal.ndim, 2), dtype='int')
        pad_width[self.axis, 1] = (
            self.frame_duration
            - (signal.shape[self.axis] % self.frame_duration)
        )
        signal = np.pad(signal, pad_width)
        frames = np.array_split(
            signal, ceil(signal.shape[self.axis] / self.frame_duration),
            axis=self.axis
        )
        # https://github.com/wiseman/py-webrtcvad/blob/master/example.py
        ring_buffer = deque(maxlen=self.padding_duration // self.frame_duration)
        triggered = False
        activity_mask = np.zeros(shape[self.axis], dtype='bool')
        if self.reset:
            # Ensure deterministic results
            self.vad = webrtcvad.Vad(self.mode)
        for i, frame in enumerate(frames):
            # Convert frame to PCM
            frame = (frame * 2**15).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            if not triggered:
                ring_buffer.append((i, frame, is_speech))
                num_voiced = len([f for _, f, speech in ring_buffer if speech])
                if num_voiced > self.threshold * ring_buffer.maxlen:
                    triggered = True
                    for j, f, _ in ring_buffer:
                        activity_mask[
                            j*self.frame_duration:(j+1)*self.frame_duration
                        ] = True
                    ring_buffer.clear()
            else:
                activity_mask[
                    i*self.frame_duration:(i+1)*self.frame_duration] = True
                ring_buffer.append((i, frame, is_speech))
                num_unvoiced = len([
                    f for _, f, speech in ring_buffer if not speech])
                if num_unvoiced > self.threshold * ring_buffer.maxlen:
                    triggered = False
                    ring_buffer.clear()
        return ArrayInterval(activity_mask)
