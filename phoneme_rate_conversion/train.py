# pylint: disable=E1120
from pathlib import Path
import os
from math import ceil, log2
import inspect

import sacred
from sacred import Experiment, commands
from sacred.utils import ConfigError
from sacred.observers import FileStorageObserver
import torch
from torch import nn

import paderbox as pb

import padertorch as pt
from padertorch.contrib.je.modules.conv import CNN1d

from .database import Timit, LibriSpeech, Database, ConcatDatabase
from .data import get_training_data
from .utils.mappers import DynamicTimeSeriesBatch
from .modules.rnn import RCNN1d
from .modules.time_scaling import LinearIndicesSampler
from .model import EstimateLocalSpeechRate

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False
NICKNAME = 'speech_rate/train'
ex = Experiment(NICKNAME, save_git_info=False)

JSONS_ROOT = Path(
    *Path(inspect.getframeinfo(inspect.currentframe()).filename).parts[:-2]
) / 'jsons'
DB_ROOT = Path(os.environ['DB_ROOT'])


def get_alignment_key_list(load_phones, load_syllables):
    return [
        k for k, loaded
        in zip(['phones', 'syllables'], [load_phones, load_syllables])
        if loaded]


def get_stft_args(
    audio_reader, shift_in_ms=12.5, window_length_in_ms=50.0, window='hann',
    **kwargs
):
    def ms_to_samples(size_in_ms):
        return int(size_in_ms * audio_reader['target_sample_rate'] / 1000)

    return {
        'shift': ms_to_samples(shift_in_ms),
        'window_length': ms_to_samples(window_length_in_ms),
        'size': 2 ** ceil(log2(ms_to_samples(window_length_in_ms))),
        'window': window,
        **kwargs,
    }


@ex.config
def defaults():
    test_run = False
    device = None
    resume = False

    database_name = 'librispeech'
    storage_dir = pt.io.get_new_storage_dir(
        'speech_rate', id_naming='index', mkdir=not test_run,
        prefix=database_name,
    )

    batcher = DynamicTimeSeriesBatch.get_config(updates=dict(
        batch_size=32, max_padding_rate=.2, expiration=4000))

    window_size_ms = 50
    shift_ms = 12.5
    load_phones = True
    phone_ali_root = DB_ROOT / 'librispeech_phone_ali'
    time_warp = True
    source_sample_rate = 16_000
    target_sample_rate = 16_000
    if database_name == 'timit':
        data = Timit(
            json_path=JSONS_ROOT / 'timit.json',
            path_prefix=DB_ROOT,
            dataset_names=['train', 'test'],
        ).to_dict()
    elif database_name == 'librispeech':
        data = LibriSpeech(
            json_path=JSONS_ROOT / 'librispeech.json',
            path_prefix=DB_ROOT,
            dataset_names=['train_clean_100', 'train_clean_360', 'dev_clean'],
            portions=[[.6, .0], [1., 0.], [0., 1.]],
        ).to_dict()
    elif database_name == 'libritimit':
        data = [
            (
                'timit', Timit(
                    json_path=JSONS_ROOT / 'timit.json',
                    path_prefix=DB_ROOT,
                    dataset_names=['train', 'test'],
                ).to_dict()
            ),
            (
                'librispeech', LibriSpeech(
                    json_path=JSONS_ROOT / 'librispeech.json',
                    path_prefix=DB_ROOT,
                    dataset_names=[
                        'train_clean_100', 'train_clean_360', 'dev_clean'],
                    portions=[[.6, .0], [1., 0.], [0., 1.]],
                ).to_dict()
            )
        ]
    else:
        raise ConfigError(f'Invalid database_name {database_name}')
    audio_reader = {
        'source_sample_rate': source_sample_rate,
        'target_sample_rate': target_sample_rate,
        'alignment_keys': get_alignment_key_list(load_phones, False),
    }
    stft = get_stft_args(
        audio_reader, shift_ms, window_size_ms,
        alignment_keys=get_alignment_key_list(load_phones, False)
    )

    in_channels = 3
    number_of_filters = 80
    width = 256
    out_channels = 128
    hidden_size = 128
    pad_type = 'both'
    rnn = True
    l2_norm = False
    trainer = {
        'storage_dir': storage_dir,
        'optimizer': {
            'factory': pt.optimizer.Adam,
            'lr': 5e-4,
            'gradient_clipping': 20.,
        },
        'summary_trigger': (1000, 'iteration'),
        'checkpoint_trigger': (1, 'epoch'),
        'stop_trigger': (200_000, 'iteration'),
        'loss_weights': {'mse': 1.},
        'model': {
            'factory': EstimateLocalSpeechRate,
            'label_key': '+'.join(get_alignment_key_list(load_phones, False)),
            'l2_norm': l2_norm,
            'feature_extractor': {
                'sample_rate': audio_reader['target_sample_rate'],
                'stft_size': stft['size'],
                'number_of_filters': number_of_filters,
                'lowest_frequency': 80,
                'highest_frequency': 7600,
                'htk_mel': False,
                'add_deltas': True,
                'add_delta_deltas': True,
                'norm_statistics_axis': 'bt',
            },
            'lsr_extractor': {
                'rate': audio_reader['target_sample_rate'] // stft['shift'],
                'window_size': int(
                    audio_reader['target_sample_rate'] / stft['shift'] * 0.625
                ),  # window of 625 ms
                'shift': 1,
                'pad_type': pad_type,
            },
            'context_net': {
                'pad_type': pad_type,
                'activation_fn': ['identity'] + ['leaky_relu'] * 5 + ['relu'],
                'norm': 'batch',
                'dropout': .0,
                'output_layer': False,
            },
        },
    }
    if l2_norm:
        trainer['loss_weights']['l2_penalty'] = 1.
    if time_warp:
        trainer['model']['feature_extractor'].update({
            'time_warping_fn': {
                'factory': LinearIndicesSampler,
                'sampling_fn': {
                    'factory': pb.utils.random_utils.Uniform,
                    'low': 0.7,
                    'high': 1.3,
                }
            },
        })
    trainer['model']['context_net'].update({
        # time receptive field: 625 ms
        'factory': CNN1d,
        'in_channels': in_channels * number_of_filters,
        'out_channels': [width] * 5 + [out_channels],
        'kernel_size': [3] + 5 * [5],
        'dilation': [1, 2, 2, 1, 1, 1],
        'residual_connections': [None, 3] + 4 * [None],
        'stride': 1,
    })
    if pad_type is None:
        trainer['model']['context_net'].update({
            'out_channels': [width] * 4 + [out_channels],
            'kernel_size': 5 * [
                int(
                    audio_reader['target_sample_rate']
                    / stft['shift'] * 0.625) // 5 + 1
            ],
            'dilation': 1,
            'residual_connections': None,
        })
    if rnn:
        trainer['model']['rnn'] = {
            'factory': RCNN1d,
            'rnn': {
                'factory': nn.LSTM,
                'batch_first': True,
                'bidirectional': False,
                'num_layers': 2,
                'input_size': out_channels,
                'hidden_size': hidden_size,
            },
            'cnn': {
                'factory': CNN1d,
                'in_channels': hidden_size,
                'out_channels': [1],
                'kernel_size': 5,
                'output_layer': False,
                'activation_fn': {'factory': nn.Softplus},
                'pad_type': 'both',
                'norm': None,
            },
        }
    else:
        trainer['model']['rnn'] = None

    pt.Trainer.get_config(trainer)

    if not test_run:
        ex.observers.append(FileStorageObserver(Path(storage_dir) / 'sacred'))


@ex.capture
def get_iterables(data, phone_ali_root, audio_reader, stft, batcher):
    if isinstance(data, dict):
        data = Database(**data)
    else:
        data = ConcatDatabase(*data)
    batcher = DynamicTimeSeriesBatch.from_config(batcher)
    training_set, validation_set = get_training_data(
        data, audio_reader, stft, phone_ali_root,
        batcher,
        is_train_set=[True, False],
        shuffle=True,
    )
    return training_set, validation_set


@ex.automain
def main(
    _run, _config, trainer, resume, test_run, device,
):
    commands.print_config(_run)

    trainer = pt.Trainer.from_config(trainer)

    train_iter, valid_iter = get_iterables()

    if test_run:
        trainer.test_run(train_iter, valid_iter, device=device)
    else:
        if not resume:
            Path(trainer.storage_dir).mkdir(parents=True, exist_ok=True)
            pt.io.dump_config(
                pt.configurable.recursive_class_to_str(_config),
                Path(trainer.storage_dir) / 'config.yaml',
            )
        trainer.register_validation_hook(
            valid_iter, metric='rel_diff_phones', maximize=False)
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
        trainer.train(train_iter, resume=resume, device=device)
