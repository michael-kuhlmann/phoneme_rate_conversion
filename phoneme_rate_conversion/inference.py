from typing import Union, Tuple
from pathlib import Path
import abc
import argparse
import dill

import numpy as np
import torch
import paderbox as pb
from paderbox.transform.module_stft import STFT
from paderbox.transform.module_resample import resample_sox
import padertorch as pt

from .model import EstimateLocalSpeechRate
from .modules.time_scaling import WSOLA
from .utils.vad import WebRTCVAD


# TODO: Readme


class _Converter(pt.Configurable, abc.ABC):
    sample_rate: int
    time_scale_fn: callable
    stft: Union[STFT, None]

    @staticmethod
    def _check_wav(wav):
        if wav.ndim == 2:
            if wav.shape[0] > 1:
                raise ValueError(
                    'Expected mono signal with shape (T,) or (1,T) but got '
                    f'{wav.ndim}-dim signal with shape {wav.shape}'
                )

    @staticmethod
    def _resample(wav, *, in_rate, out_rate):
        wav = wav.astype(np.float32)
        if in_rate != out_rate:
            if wav.ndim == 2:
                wav = np.stack([
                    resample_sox(channel, in_rate=in_rate, out_rate=out_rate)
                    for channel in wav
                ])
            else:
                wav = resample_sox(wav, in_rate=in_rate, out_rate=out_rate)
        return wav

    def __call__(
        self, content_wav: np.ndarray, style_wav: np.ndarray,
        *,
        in_rate: int = 16_000,
    ) -> np.ndarray:
        """
        Imprint `style_wav` speaking rate onto `content_wav`.

        Args:
            content_wav: Mono audio signal. Shape (N,) or (1, N)
            style_wav: Mono audio signal. Shape (M,) or (1, M)
            in_rate: Sampling rate of `content_wav` and `style_wav`
        """
        factor = self.get_interpolation_factor(
            content_wav, style_wav, rate=in_rate)
        wav, *_ = self.time_scale_fn(
            self._resample(
                content_wav.squeeze(), in_rate=in_rate,
                out_rate=self.sample_rate
            ), factor
        )
        return self._resample(wav, in_rate=self.sample_rate, out_rate=in_rate)

    def wav_to_stft(self, wav: np.ndarray, rate: int = 16_000) -> np.ndarray:
        """
        Extract STFT signal from audio signal.

        Args:
            wav: Audio signal of shape (T,) or (1, T) where T: number of samples
            rate: Sampling rate of `wav`

        Returns:
            STFT signal
        """
        if self.stft is None:
            raise TypeError(f'{self.__class__} does not support STFT')
        self._check_wav(wav)
        if rate != self.sample_rate:
            wav = self._resample(
                wav, in_rate=rate, out_rate=self.sample_rate)
        stft = self.stft(wav)  # (#frames, #bins)
        return np.stack([stft.real, stft.imag], axis=-1).astype('float32')

    @abc.abstractmethod
    def get_interpolation_factor(
            self, content_wav: np.ndarray, style_wav: np.ndarray,
            rate: int = 16_000,
            content_stft: Union[np.ndarray, None] = None,
            style_stft: Union[np.ndarray, None] = None,
    ) -> float:
        pass


class SpeakingRateConverter(_Converter):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        model_config = pt.io.load_config(
            Path(config['model_dir']) / config['config_name'])
        config['time_scale_fn'] = {
            'factory': WSOLA,
            'sample_rate': model_config['audio_reader']['target_sample_rate'],
        }
        config['vad'] = {
            'factory': WebRTCVAD,
            'sample_rate': model_config['audio_reader']['target_sample_rate'],
        }

    def __init__(
        self, model_dir: Union[str, Path],
        time_scale_fn: callable,
        vad: Union[callable, None] = None,
        checkpoint_name='scenario1_libritimit_best.pth',
        *,
        device: Union[str, int] = 'cpu',
        config_name='config.yaml',
    ):
        self.speech_rate_estimator = EstimateLocalSpeechRate.from_storage_dir(
            model_dir, config_name,
            checkpoint_name=checkpoint_name,
        ).to(device).eval()
        self.time_scale_fn = time_scale_fn
        self.vad = vad
        model_config = pt.io.load_config(
            Path(model_dir) / config_name)
        self.sample_rate = model_config['audio_reader']['target_sample_rate']
        stft_params = model_config['stft']
        stft_params.pop('alignment_keys', None)
        self.stft = STFT(**stft_params)

    @staticmethod
    def _maybe_expand_stft(_stft: np.ndarray) -> np.ndarray:
        # Expected shape: (B, C, T, F, 2)
        for _ in range(max(0, 5 - _stft.ndim)):
            _stft = _stft[None]
        return _stft

    def estimate_speech_rate_from_wav(
        self, wav: np.ndarray, rate: int = 16_000,
        stft_signal: Union[np.ndarray, None] = None
    ) -> Tuple[int, int]:
        self._check_wav(wav)

        target_sample_rate = (
            self.speech_rate_estimator.feature_extractor.mel_transform
                .sample_rate
        )
        if rate != target_sample_rate:
            wav = pb.transform.module_resample.resample(
                wav, in_rate=rate, out_rate=target_sample_rate)
        if stft_signal is None:
            stft_signal = self.wav_to_stft(wav, rate=target_sample_rate)
        # Estimate local speech rate
        lsr = self.speech_rate_estimator.predict(
            self.speech_rate_estimator.example_to_device(
                self._maybe_expand_stft(stft_signal)
            )
        )
        # Compute number of acoustic units
        units = self.speech_rate_estimator.lsr_extractor.get_num_segments(lsr)\
            .item()
        # Compute number of non-silence samples
        if wav.ndim == 2:
            wav = wav.squeeze(0)
        if self.vad is not None:
            num_samples = self.vad(
                self._resample(
                    wav, in_rate=target_sample_rate, out_rate=self.sample_rate)
            ).sum().item()
        else:
            num_samples = wav.shape[-1]
        return units, num_samples

    def get_interpolation_factor(
        self, content_wav: np.ndarray, style_wav: np.ndarray,
        rate: int = 16_000,
        content_stft: Union[np.ndarray, None] = None,
        style_stft: Union[np.ndarray, None] = None,
    ) -> float:
        content_units, content_samples = self.estimate_speech_rate_from_wav(
            content_wav, rate=rate, stft_signal=content_stft)
        style_units, style_samples = self.estimate_speech_rate_from_wav(
            style_wav, rate=rate, stft_signal=style_stft)
        # Compute ratio of estimated speech rates
        factor = style_units * content_samples / (content_units * style_samples)
        return factor


class UnsupSegSpeakingRateConverter(_Converter):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['time_scale_fn'] = {
            'factory': WSOLA,
            'sample_rate': 16_000,
        }
        config['vad'] = {
            'factory': WebRTCVAD,
            'sample_rate': 16_000,
        }

    def __init__(
        self,
        model_dir: Union[str, Path],
        time_scale_fn: callable,
        vad: Union[callable, None] = None,
        checkpoint_name='scenario2_libritimit_best.ckpt',
        *,
        device: Union[str, int] = 'cpu',
    ):
        self.model_dir = model_dir
        self.time_scale_fn = time_scale_fn
        self.vad = vad
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.sample_rate = 16_000
        self.stft = None
        self._load_segmenter()

    def _load_segmenter(self):
        try:
            from unsup_seg.next_frame_classifier import NextFrameClassifier
        except ImportError as e:
            raise ImportError(
                'Please install unsup_seg '
                '(https://github.com/michael-kuhlmann/UnsupSeg.git) or '
                'activate the correct environment'
            ) from e
        ckpt = Path(self.model_dir) / 'checkpoints' / self.checkpoint_name
        ckpt = torch.load(ckpt, map_location='cpu')
        hp = argparse.Namespace(**dict(ckpt['hparams']))
        # Load weights and peak detection params
        self.segmenter = NextFrameClassifier(hp)
        weights = ckpt["state_dict"]
        weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
        self.segmenter.load_state_dict(weights)
        self.segmenter.eval().to(self.device)
        self.peak_detection_params = dill.loads(ckpt['peak_detection_params'])[
            'cpc_1']

    def estimate_speech_rate_from_wav(
            self, wav: np.ndarray, rate: int = 16_000) -> Tuple[int, int]:
        from unsup_seg.utils import (
            detect_peaks, max_min_norm, replicate_first_k_frames)

        if wav.ndim == 2:
            assert wav.shape[0] == 1, wav.shape
            wav = wav.squeeze(0)
        if rate != 16_000:
            wav = pb.transform.module_resample.resample(
                wav, in_rate=rate, out_rate=16_000)
        if self.vad is not None:
            vad_mask = self.vad(wav)
            num_samples = vad_mask.sum()
            wav = wav[vad_mask[:]]
        else:
            num_samples = wav.shape[0]
        x = torch.from_numpy(wav).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            preds = self.segmenter(x)
            preds = preds[1][0].cpu()  # Get scores of positive pair
            preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
            preds = 1 - max_min_norm(
                preds)  # normalize scores (good for visualizations)
            preds = detect_peaks(
                x=preds,
                lengths=[preds.shape[1]],
                prominence=self.peak_detection_params["prominence"],
                width=self.peak_detection_params["width"],
                distance=self.peak_detection_params["distance"]
            )[0]  # Run peak detection on scores
        return preds.shape[-1] + 1, num_samples

    def get_interpolation_factor(
        self, content_wav: np.ndarray, style_wav: np.ndarray,
        rate: int = 16_000, **kwargs,
    ) -> float:
        content_units, content_samples = self.estimate_speech_rate_from_wav(
            content_wav, rate=rate)
        style_units, style_samples = self.estimate_speech_rate_from_wav(
            style_wav, rate=rate)
        # Compute ratio of estimated speech rates
        factor = style_units * content_samples / (content_units * style_samples)
        return factor
