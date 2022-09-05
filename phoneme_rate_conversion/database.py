import dataclasses
from dataclasses import asdict
from typing import Union, List, Tuple
from pathlib import Path
from math import ceil
from collections import defaultdict

from cached_property import cached_property
import numpy as np
import lazy_dataset
from lazy_dataset.database import JsonDatabase
import paderbox as pb
from paderbox.utils.misc import interleave
from padertorch.utils import to_list


@dataclasses.dataclass
class _Database:
    json_path: Union[str, Path]


@dataclasses.dataclass
class _DatabaseDefaults:
    database_name: str = ''
    dataset_names: Union[str, List[str]] = ''
    source_sample_rate: int = 0
    path_prefix: [str, Path, None] = None
    audio_key: [str, None] = 'observation'
    portions: Union[List[List[float]], List[float], Tuple[float], None] = None
    n_speakers: Union[List[int], int, None] = None
    speaker_onset: Union[List[int], int, None] = None
    seed: int = 0
    split_type: str = 'speaker_wise'
    stratify_gender: bool = False
    storage_dir: Union[str, Path, None] = None


@dataclasses.dataclass
class Database(_DatabaseDefaults, _Database):
    def __post_init__(self):
        if len(self.dataset_names) == 0:
            raise TypeError("Missing required argument 'dataset_names'")
        if len(self.database_name) == 0:
            raise TypeError("Missing required argument 'database_name'")
        if self.source_sample_rate == 0:
            raise TypeError("Missing required argument 'source_sample_rate'")
        if isinstance(self.dataset_names, str):
            self.dataset_names = [self.dataset_names]
        if isinstance(self.path_prefix, str):
            self.path_prefix = Path(self.path_prefix)
        if self.portions is None:
            self.portions = np.eye(len(self.dataset_names)).tolist()
        else:
            self.portions = to_list(self.portions, len(self.dataset_names))
        self.n_speakers = to_list(self.n_speakers, len(self.dataset_names))
        self.speaker_onset = to_list(
            self.speaker_onset, len(self.dataset_names))
        if isinstance(self.storage_dir, str):
            self.storage_dir = Path(self.storage_dir)

    @cached_property
    def db(self):
        if self.json_path is not None:
            return JsonDatabase(self.json_path)
        raise TypeError('Missing an argument for json_path')

    def __call__(self, example):
        return self._prepare(example)

    def _prepare(self, example):
        if self.audio_key is not None and isinstance(
                example['audio_path'], dict):
            example['audio_path'] = example['audio_path'][self.audio_key]
        if self.path_prefix is not None:
            example = pb.utils.nested.flatten(example)
            for key, value in example.items():
                if 'audio_path' in key:
                    example[key] = self.path_prefix / value
            example = pb.utils.nested.deflatten(example)
        return example

    def _finalize_dataset(self, dataset: lazy_dataset.Dataset):
        return dataset

    def get_datasets(self):
        def _add_database_name(example):
            example['database'] = self.database_name
            return example

        datasets = []
        for i, ds_name in enumerate(self.dataset_names):
            rng = np.random.RandomState(self.seed)
            dataset_i = self.db.get_dataset(ds_name)
            speaker_ids = sorted({(
                ex['speaker_id'], ex['gender']) for ex in dataset_i},
                key=lambda x: x[0],
            )
            rng.shuffle(speaker_ids)
            speaker_onset = self.speaker_onset[i]
            n_speakers = self.n_speakers[i]
            if speaker_onset is None:
                speaker_onset = 0
            if self.stratify_gender:
                male_speakers = list(filter(
                    lambda x: x[1] == 'male', speaker_ids))
                female_speakers = list(filter(
                    lambda x: x[1] == 'female', speaker_ids))
                l1 = (
                    male_speakers if len(male_speakers) >= len(female_speakers)
                    else female_speakers
                )
                l2 = (
                    male_speakers if len(male_speakers) < len(female_speakers)
                    else female_speakers
                )
                imbalance_ratio = len(l1) / len(l2)
                if imbalance_ratio >= 2:
                    sub_len = ceil(len(l1) / imbalance_ratio)
                    l1 = [
                        l1[i*sub_len:(i+1)*sub_len]
                        for i in range(ceil(len(l1)/sub_len))
                    ]
                else:
                    l1 = [l1]
                # End of list may be closer to `imbalance_ratio` than beginning
                # Better balance is more desirable for test than train split
                speaker_ids = reversed(list(interleave(*l1, l2)))
            speaker_ids = list(map(lambda x: x[0], speaker_ids))
            speaker_ids = speaker_ids[speaker_onset:][:n_speakers]
            if self.storage_dir is not None and self.storage_dir.exists():
                dump = {
                    'speaker_ids': sorted(speaker_ids),
                    'speaker_onset': speaker_onset,
                    'n_speakers': n_speakers,
                    'seed': self.seed,
                    'stratify_gender': self.stratify_gender,
                }
                pb.io.dump_json(
                    dump, self.storage_dir / (
                        f'speaker_id-{ds_name}-speaker_onset={speaker_onset}'
                        f'-n_speakers={n_speakers}.json'
                    )
                )
            dataset_i = dataset_i.filter(
                lambda ex: ex['speaker_id'] in speaker_ids, lazy=False)
            dataset_i = self._finalize_dataset(dataset_i)
            datasets.append(dataset_i.map(_add_database_name))
        return self.split_datasets(datasets)

    def split_datasets(self, datasets):
        portioned_datasets = []
        for i, (portion, dataset) in enumerate(zip(self.portions, datasets)):
            splits = split_dataset(
                dataset, portion, seed=self.seed, split_type=self.split_type)
            for j, split in enumerate(splits):
                if isinstance(split, lazy_dataset.Dataset):
                    if len(portioned_datasets) <= j:
                        portioned_datasets.append(split)
                    else:
                        portioned_datasets[j] = lazy_dataset.concatenate(
                            (portioned_datasets[j], split))

        return tuple(portioned_datasets)

    def to_dict(self):
        return asdict(self)


@dataclasses.dataclass
class _LibriSpeechDefaults(_DatabaseDefaults):
    database_name: str = 'librispeech'
    dataset_names: Union[str, List[str]] = 'train_clean_100'
    source_sample_rate: int = 16_000


@dataclasses.dataclass
class LibriSpeech(_LibriSpeechDefaults, Database):
    pass


@dataclasses.dataclass
class _TimitDefaults(_DatabaseDefaults):
    database_name: str = 'timit'
    dataset_names: Union[str, List[str]] = 'train'
    source_sample_rate: int = 16_000
    audio_key = None


@dataclasses.dataclass
class Timit(_TimitDefaults, Database):
    def _prepare(self, example: dict):
        example['phones_stop_times'] = example.pop('phones_end_times')
        example['words_stop_times'] = example.pop('words_end_times')
        return super()._prepare(example)


class ConcatDatabase:
    def __init__(self, *databases):
        self.databases = {}
        source_sample_rates = []
        self.portions = {}
        for key, database in databases:
            if isinstance(database, dict):
                database = Database(**database)
            self.databases[key] = database
            self.portions[key] = database.portions
            source_sample_rates.append(database.source_sample_rate)
        if not all([
            source_rate == source_sample_rates[0]
            for source_rate in source_sample_rates
        ]):
            raise ValueError(
                'Databases with different sample rates are not supported!')
        self.source_sample_rate = source_sample_rates[0]

    def __call__(self, example):
        return self.databases.get(example['database'], lambda x: x)(example)

    def get_datasets(self):
        datasets = []
        for database in self.databases.values():
            datasets.append(database.get_datasets())
        lens = [len(splits) for splits in datasets]
        if not all([len_i == lens[0] for len_i in lens]):
            raise RuntimeError(
                'Splits in the databases do not match:\n'
                '\n'.join([
                    f'{db_key} portions: {db.portions}'
                    for db_key, db in self.databases.items()
                ])
            )
        concat_datasets = []
        for splits in zip(*datasets):
            concat_datasets.append(lazy_dataset.concatenate(*splits))
        return concat_datasets


def split_dataset(dataset, portions, split_type='speaker_wise', seed=0):
    """Splits the utterances of each speaker in a dataset (or in a list of
    datasets).

    Args:
        dataset: a lazy dataset or a list of lazy datasets
        portions: a list of ratios stating how to split. If, e.g.,
            portions=[.6, .4] a 60%/40% split is performed and 2 datasets are
            returned. If dataset is a list you can also state separate portions
            for each dataset. If dataset is, e.g., a list of len 2, than you
            could use portions=[[.6,.4], [.5,.5]] to perform a 60%/40% split on
            the first dataset but a 50%/50% on the second dataset. Note that
            all elements in portions must >= 0 and the sum over the last dim of
            portions must be <= 1 .
        seed: seed for random splitting

    Returns:

    """
    if np.isscalar(portions):
        portions = [portions]
    if isinstance(dataset, (list, tuple)):
        if isinstance(portions[0], (list, tuple)):
            assert len(portions) == len(dataset), (len(portions), len(dataset))
            assert len({len(p) for p in portions}) == 1
        else:
            portions = len(dataset) * [portions]
        n = len(portions[0])
        subsets = [[] for _ in range(n)]
        for ds, p in zip(dataset, portions):
            ds_split = split_dataset(ds, p, split_type=split_type, seed=seed)
            for i in range(n):
                if len(ds_split[i]) > 0:
                    subsets[i].append(ds_split[i])
        for i, ds in enumerate(subsets):
            if len(ds) > 0:
                subsets[i] = lazy_dataset.concatenate(ds)
        return subsets

    assert (np.array(portions) >= 0).all(), portions
    assert sum(portions) <= 1, portions
    if all(np.array(portions) == 0.):
        return [[] for _ in portions]
    if any(np.array(portions) == 1.):
        return [dataset if portion == 1. else [] for portion in portions]
    portions = np.cumsum(portions)

    portion_indices = [[] for _ in range(len(portions))]

    def _split_and_append_indices(indices_):
        np.random.RandomState(seed).shuffle(indices_)
        indices_split_ = np.split(
            indices_,
            np.round((len(indices_) * np.array(portions))).astype(np.int64)
        )
        for i in range(len(indices_split_)-1):
            if len(indices_split_[i]) > 0:
                portion_indices[i].extend(indices_split_[i].tolist())

    if split_type == 'speaker_wise':
        speaker_dist = defaultdict(list)
        for i, ex in enumerate(dataset):
            spk_id = ex['speaker_id']
            if isinstance(spk_id, np.ndarray):
                spk_id = int(spk_id)
            speaker_dist[spk_id].append(i)
        portion_indices = [[] for _ in range(len(portions))]
        for indices in speaker_dist.values():
            _split_and_append_indices(indices)
    elif split_type == 'random':
        _split_and_append_indices(np.arange(len(dataset)))
    else:
        raise ValueError(f'Invalid split_type {split_type}')

    return [
        dataset[sorted(indices)] if len(indices) > 0 else []
        for indices in portion_indices
    ]
