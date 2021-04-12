import argparse
import os
import random
from pathlib import Path
from zipfile import ZipFile
from num2words import num2words
from typing import Optional
from dataclasses import dataclass
import json
import re


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
    bitrate: Optional[int] = ""
    format: Optional[str] = "wav"
    zip_name: Optional[str] = ""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.05,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    # parser.add_argument(
    #     "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    # )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = Path(args.root)
    rand = random.Random(args.seed)

    lexicon = set()
    dict = {}

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f, \
            open(os.path.join(args.dest, "valid.tsv"), "w") as valid_f, \
            open(os.path.join(args.dest, "train.ltr"), "w") as train_ltr, \
            open(os.path.join(args.dest, "valid.ltr"), "w") as valid_ltr, \
            open(os.path.join(args.dest, "train.wrd"), "w") as train_wrd, \
            open(os.path.join(args.dest, "valid.wrd"), "w") as valid_wrd:

        print(dir_path, file=train_f)
        print(dir_path, file=valid_f)

        for zip_file in list(dir_path.glob('**/*.zip')):
            with ZipFile(zip_file) as myzip:
                with myzip.open("manifest.jsona") as lines:
                    for line in lines:
                        entry = json.loads(line)
                        metadata = AudioRecordMetadata(**entry)
                        metadata.text = complete_normalize(metadata.text)

                        if args.path_must_contain and args.path_must_contain not in zip_file:
                            continue
                        else:
                            if rand.random() > args.valid_percent:
                                dest = train_f
                                wrd_out = train_wrd
                                ltr_out = train_ltr
                            else:
                                dest = valid_f
                                wrd_out = valid_wrd
                                ltr_out = valid_ltr

                            print("{}\t{}\t{}".format(
                                os.path.relpath(zip_file, dir_path),
                                metadata.zip_entry_name,
                                metadata.sample_count),
                                file=dest)

                            print(metadata.text, file=wrd_out)
                            ltr_output = " ".join(list(metadata.text.replace(" ", "|"))) + " |"
                            print(ltr_output, file=ltr_out)

                            lexicon.update(metadata.text.split())

                            for c in ltr_output.split():
                                if c not in dict.keys():
                                    dict[c] = 0
                                dict[c] += 1
                            # Add | at the end
                            dict["|"] += 1

    with open(os.path.join(args.dest, "lexicon.txt"), "w") as lex_out, \
            open(os.path.join(args.dest, "dict.ltr.txt"), "w") as dict_out:

        for c, n in sorted(dict.items(), key=lambda item: item[1], reverse=True):
            print("{} {}".format(c, n), file=dict_out)

        for word in lexicon:
            print("{} {} |".format(word, " ".join(word)), file=lex_out)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
