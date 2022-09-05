from pathlib import Path
from typing import Union
import re
import dataclasses
import warnings
import psutil

import numpy as np
import textgrids
import lazy_dataset
from lazy_dataset import FilterException
from paderbox.utils.nested import nested_op
import padertorch as pt
from padertorch.contrib.je.data.transforms import Collate


@dataclasses.dataclass
class TextGridAlignmentReader:
    """
    Load alignments (phone, syllables, ...) from Praat TextGrid files.
    `ali_root` must adhere to following structure:
        ali_root
        |- <speaker_id_1>
        |   |- <example_id_1_1>.TextGrid
        |   |- <example_id_1_2>.TextGrid
        |   ...
        |- <speaker_id_2>
        |   |- <example_id_2_1>.TextGrid
        |   ...
        ...
    """
    ali_root: Union[str, Path]
    label_key: str
    to_array: bool = False
    reduce_labels: bool = True
    verbose: bool = False
    drop_silence: bool = False
    silence_label: Union[str, list] = dataclasses.field(
        default_factory=lambda: ['h#', 'pau', ''])

    def __post_init__(self):
        self.ali_root = Path(self.ali_root)

    def filter_fn(self, example):
        example_id = example['example_id']
        speaker_id = example['speaker_id']
        return (self.ali_root / speaker_id / example_id).with_suffix(
            '.TextGrid').exists()

    def __call__(self, example: dict):
        """
        Arguments:
            example:

        Raises:
            lazy_dataset.FilterException if no alignment file for the example is
            found.
        """
        example_id = example['example_id']
        speaker_id = example['speaker_id']

        try:
            grid = textgrids.TextGrid(
                (self.ali_root / speaker_id / example_id).with_suffix(
                    '.TextGrid')
            )
            start_times = []
            stop_times = []
            labels = []
            for interval in grid[self.label_key]:
                if isinstance(interval, textgrids.Point):
                    raise TypeError(
                        'PoinTier is not supported. Convert it to an '
                        'IntervalTier'
                    )
                text = interval.text
                if self.drop_silence and text in self.silence_label:
                    continue
                start_times.append(interval.xmin)
                stop_times.append(interval.xmax)
                if self.reduce_labels:
                    # remove suffix digit classifiers from labels
                    text = re.sub(r'\d', '', text)
                labels.append(text)
            if self.to_array:
                start_times = np.array(start_times)
                stop_times = np.array(stop_times)
                labels = np.array(labels)
            example[f'{self.label_key}_start_times'] = start_times
            example[f'{self.label_key}_stop_times'] = stop_times
            example[self.label_key] = labels
            return example
        except FileNotFoundError as exc:
            if self.verbose:
                warnings.warn(f'No alignment for {example_id}. Drop example.')
            raise FilterException() from exc


@dataclasses.dataclass
class DynamicBatch(pt.Configurable):
    batch_size: Union[int, None]
    bucket_cls: str
    expiration: int = 2000
    drop_incomplete: bool = False
    sort_key: str = 'seq_len'
    reverse_sort: bool = True

    def __post_init__(self):
        if self.batch_size is not None:
            self.expiration = self.expiration * self.batch_size

    def __call__(self, dataset: lazy_dataset.Dataset, **kwargs):
        bucket_cls = pt.configurable.import_class(self.bucket_cls)
        if self.batch_size is None:
            return dataset
        if self.batch_size == 1:
            return (
                dataset
                .batch(1, drop_last=self.drop_incomplete)
                .map(Collate(StackArrays()))
            )
        return (
            dataset
            .batch_dynamic_bucket(
                bucket_cls,
                batch_size=self.batch_size,
                expiration=self.expiration,
                drop_incomplete=self.drop_incomplete,
                sort_key=self.sort_key,
                reverse_sort=self.reverse_sort,
                **kwargs
            )
            .map(Collate(StackArrays()))
        )


@dataclasses.dataclass
class DynamicTimeSeriesBatch(DynamicBatch):
    bucket_cls: str = pt.configurable.class_to_str(
        lazy_dataset.core.DynamicTimeSeriesBucket)
    len_key: str = 'seq_len'
    max_padding_rate: float = 0.05
    max_total_size: Union[None, int] = None

    def __call__(self, dataset: lazy_dataset.Dataset, **kwargs):
        return super().__call__(
            dataset,
            len_key=self.len_key,
            max_padding_rate=self.max_padding_rate,
            max_total_size=self.max_total_size,
            **kwargs
        )


@dataclasses.dataclass
class StackArrays:

    def __call__(self, example):
        if isinstance(example, dict):
            example = nested_op(self.stack, example, sequence_type=())
        elif isinstance(example, (list, tuple)):
            example = self.stack(example)
        return example

    def stack(self, batch):
        if (
            isinstance(batch, (list, tuple))
            and isinstance(batch[0], np.ndarray)
        ):
            ndim = batch[0].ndim

            shapes = [a.shape for a in batch]
            target_shape = [
                max([shape[i] for shape in shapes]) for i in range(ndim)
            ]
            stacked_arrays = []
            for a in batch:
                a_padded = np.zeros(target_shape, dtype=a.dtype)
                indices = tuple([slice(s) for s in a.shape])
                a_padded[indices] = a
                stacked_arrays.append(a_padded)
            return np.stack(stacked_arrays)
        return batch


@dataclasses.dataclass
class Prefetch:
    batch_size: int
    is_train_set: bool = True
    shuffle: bool = True
    buffer_size: int = None
    num_workers: int = None
    rng: Union[np.random.RandomState, None] = None
    catch_filter_exception: bool = True

    def __post_init__(self):
        if self.buffer_size is None:
            self.buffer_size = 2 * self.batch_size
        if self.num_workers is None:
            self.num_workers = max(len(psutil.Process().cpu_affinity()) - 2, 1)
        self.num_workers = min(self.buffer_size, self.num_workers)

    def __call__(self, dataset: lazy_dataset.Dataset):
        if not isinstance(dataset, lazy_dataset.Dataset):
            raise TypeError(
                'Prefetch expects a lazy_dataset.Dataset instance as input. '
                'Try to call Prefetch with .apply()'
            )
        if self.shuffle:
            dataset = dataset.shuffle(rng=self.rng, reshuffle=self.is_train_set)
        return dataset.prefetch(
            self.num_workers, buffer_size=self.buffer_size,
            catch_filter_exception=self.catch_filter_exception
        )
