# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .. import FairseqDataset

from zipfile import ZipFile
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO


logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
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
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if self.sample_rate != curr_sample_rate:
            wav_tensor = feats.clone().detach()
            wav_tensor = Resample(curr_sample_rate, self.sample_rate)(wav_tensor)
            feats = wav_tensor.numpy()

        if feats.dim() == 2:
            feats = feats.mean(-1)

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
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )

        self.fnames = []
        self.line_inds = set()

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                # Format: zip_file, file_name, sample_count
                assert len(items) == 3, line
                sz = int(items[-1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    continue
                # Format: (zip_file, file_name)
                self.fnames.append((items[0], items[1]))
                self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        zip_file = os.path.join(self.root_dir, self.fnames[index][0])
        fname = self.fnames[index][1]

        with ZipFile(zip_file) as myzip:
            with myzip.open(fname) as myfile:
                wav = myfile.read()

        if fname.endswith(".wav"):
            waveform, sample_rate = self.read_wav(wav)
        elif fname.endswith(".mp3"):
            waveform, sample_rate = self.read_mp3(wav)
        else:
            raise ValueError(f"File extension for {fname} was not recognized.")

        # wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(waveform).float()
        feats = self.postprocess(feats, sample_rate)
        return {"id": index, "source": feats}

    def read_wav(self, bytes):
        waveform, sample_rate = sf.read(BytesIO(bytes), dtype="float32")
        return waveform, sample_rate

    def read_mp3(self, bytes, normalized=False):
        """MP3 to numpy array"""
        audio = AudioSegment.from_mp3(BytesIO(bytes))
        audio = audio.set_channels(1)

        y = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if normalized:
            return y / 2 ** 15, audio.frame_rate
        else:
            return y, audio.frame_rate
