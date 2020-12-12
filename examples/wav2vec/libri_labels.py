#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import os
from pathlib import Path
from fairseq.data.audio.raw_audio_dataset import read_manifest_metadata


def read_files(path: Path):
    data_files = list(path.glob('**/*.zip'))
    data_files = filter(Path.is_file, data_files)
    return list(data_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_files = read_files(Path(args.data_dir))

    with open(os.path.join(args.output_dir, args.output_name + ".ltr"), "w") as ltr_out,\
            open(os.path.join(args.output_dir, args.output_name + ".wrd"), "w") as wrd_out:

        for data_file in data_files:
            samples = read_manifest_metadata(data_file)
            for sample in samples:
                transcription = sample.text
                print(transcription, file=wrd_out)
                print(" ".join(list(transcription.replace(" ", "|"))) + " |", file=ltr_out)


if __name__ == "__main__":
    main()
