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
import json
from zipfile import ZipFile
from pathlib import Path
import re
from num2words import num2words

from fairseq.data.audio.raw_audio_dataset import FileAudioDataset


def replace_numbers(text):
    numbers = re.findall('\d+', text)
    for number in numbers:
        num_as_word = num2words(number, lang='lt')
        text = text.replace(number, num_as_word)
    return text


def remove_special_tokens(text):
    text = text.lower()
    special_tokens = {'_pauze', '_tyla', '_ikvepimas', '_iskvepimas'}
    for tok in special_tokens:
        text = text.replace(tok.lower(), '')
    return text


non_alpha_space_pattern = re.compile(r'([^\s\w]|_)+')


def normalize_target_transcript(txt: str):
    txt = txt.lower()
    txt = txt.replace(".wav", "")
    return txt


def complete_normalize(txt: str):
    txt = normalize_target_transcript(txt)
    txt = remove_special_tokens(txt)
    txt = replace_numbers(txt)
    txt = non_alpha_space_pattern.sub("", txt)
    txt = txt.strip()
    return txt


def read_files(path: Path):
    data_files = list(path.glob('**/*.zip'))
    data_files = filter(Path.is_file, data_files)
    return list(data_files)


def read_manifest_metadata(zip_filepath):
    audio_entries = []
    with ZipFile(zip_filepath) as myzip:
        with myzip.open('manifest.jsona') as lines:
            for line in lines:
                entry = json.loads(line)
                metadata = FileAudioDataset.AudioRecordMetadata(**entry)
                metadata.text = complete_normalize(metadata.text)
                metadata.zip_name = zip_filepath
                audio_entries.append(metadata)
    return audio_entries


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
