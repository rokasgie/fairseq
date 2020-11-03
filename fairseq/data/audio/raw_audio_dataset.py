# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from io import BytesIO
import sys
from zipfile import ZipFile
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset


logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        zipped_batches,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )
        self.files = []
        self.sizes = []
        skipped = 0
        for zipped_batch in zipped_batches:
            files, sizes, skips = self.read_manifest_metadata(zipped_batch)
            skipped += skips
            self.files.extend(files)
            self.sizes.extend(sizes)
        logger.info(f"loaded {len(self.files)}, skipped {skipped} samples")

    def read_manifest_metadata(self, zip_filepath):
        files = []
        sizes = []
        skipped = 0
        with ZipFile(zip_filepath) as myzip:
            with myzip.open('manifest.jsona') as lines:
                for line in lines:
                    entry = json.loads(line)
                    metadata = FileAudioDataset.AudioRecordMetadata(**entry)
                    metadata.zip_name = zip_filepath
                    if self.min_length is not None and metadata.sample_count < self.min_length:
                        skipped += 1
                    else:
                        files.append(metadata)
                        sizes.append(metadata.sample_count)
        return files, sizes, skipped

    def read_item(self, zip_file, name):
        with ZipFile(zip_file) as myzip:
            with myzip.open(name) as myfile:
                wav = myfile.read()
                return wav

    def __getitem__(self, index):
        import soundfile as sf

        metadata: FileAudioDataset.AudioRecordMetadata = self.files[index]
        data = self.read_item(metadata.zip_name, metadata.zip_entry_name)
        wav, curr_sample_rate = sf.read(BytesIO(data))
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats}

    @dataclass()
    class AudioRecordMetadata:
        text: str
        zip_entry_name: str
        sample_rate: int
        sample_count: int
        id: str
        group: str
        batch: int
        seq_no: int
        format: Optional[str] = None
        zip_name: Optional[str] = None