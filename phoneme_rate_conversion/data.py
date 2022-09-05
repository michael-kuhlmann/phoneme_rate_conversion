from typing import Union

import numpy as np
from padertorch.utils import to_list
from padertorch.contrib.je.data.transforms import AudioReader, STFT

from .database import Database
from .utils.mappers import TextGridAlignmentReader, DynamicBatch, Prefetch


def finalize(
    example, to_array=False, alignment_keys: Union[None, list, tuple] = None
):
    example['num_samples'] = example['audio_data'].shape[-1]
    example['seq_len'] = example['stft'].shape[1]
    if alignment_keys is not None and to_array:
        for alignment_key in alignment_keys:
            example[f'{alignment_key}_start_frames'] = np.array(
                example[f'{alignment_key}_start_frames'])
            example[f'{alignment_key}_stop_frames'] = np.array(
                example[f'{alignment_key}_stop_frames'])
    example = {
        'example_id': example['example_id'],
        'num_samples': example['num_samples'],
        'seq_len': example['seq_len'],
        'audio_data': example['audio_data'],
        'stft': example['stft'],
        'phones': example['phones'],
        'phones_start_frames': example['phones_start_frames'],
        'phones_stop_frames': example['phones_stop_frames'],
    }
    return example


def get_num_samples(example):
    if 'audio_start_samples' in example:
        assert 'audio_stop_samples' in example, example.keys()
        num_samples = (
            example['audio_stop_samples'] - example['audio_start_samples'])
    else:
        num_samples = example['num_samples']
    return num_samples


def get_training_data(
    database: Database, audio_reader: dict, stft: dict,
    phone_ali_root,
    batcher: DynamicBatch, *,
    num_workers=None, is_train_set=True, shuffle=True,
    seed=0,
):
    audio_reader = AudioReader(**audio_reader)
    stft = STFT(**stft)
    datasets = database.get_datasets()
    is_train_set = to_list(is_train_set, len(datasets))
    shuffle = to_list(shuffle, len(datasets))
    batch_size = batcher.batch_size

    alignment_reader = TextGridAlignmentReader(
        phone_ali_root, 'phones', silence_label=['SIL', 'h#', 'pau', ''],
        drop_silence=True,
    )

    def _maybe_load_alignments(example):
        if example['database'] != 'timit':
            example = alignment_reader(example)
        return example

    def _maybe_filter_example(example):
        if example['database'] == 'timit':
            return True
        return alignment_reader.filter_fn(example)

    prepared_datasets = []
    for i, dataset in enumerate(datasets):
        if is_train_set[i]:
            rng = np.random
        else:
            rng = np.random.RandomState(seed)
        dataset = (
            dataset
            .map(database)
            .filter(_maybe_filter_example, lazy=False)
            .map(_maybe_load_alignments)
            .map(audio_reader)
            .map(stft)
            .map(finalize)
            .apply(Prefetch(
                batch_size, is_train_set=is_train_set[i], shuffle=shuffle[i],
                rng=rng, num_workers=num_workers,
                catch_filter_exception=False,
            ))
            .apply(batcher)
            .prefetch(1, 2, catch_filter_exception=False)
        )

        prepared_datasets.append(dataset)

    return prepared_datasets
