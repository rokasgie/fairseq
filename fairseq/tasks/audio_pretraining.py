# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from comet_ml import Experiment

import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING

from fairseq.data import AddTargetDataset, Dictionary, FileAudioDataset, encoders
from fairseq.data.data_utils import post_process
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics

import random
from pathlib import Path
from fairseq.data.audio.raw_audio_dataset import read_manifest_metadata


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class AudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    valid_perc: float = field(
        default=0.2,
        metadata={"help": "validation percentage of all data"}
    )
    sample_rate: int = field(
        default=22050,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to crop to for batching"}
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )


@register_task("audio_pretraining", dataclass=AudioPretrainingConfig)
class AudioPretrainingTask(FairseqTask):
    """"""

    cfg: AudioPretrainingConfig

    def __init__(
        self,
        cfg: AudioPretrainingConfig,
        source_dictionary=None,
        target_dictionary=None,
        dataset_files=None,
    ):
        super().__init__(cfg)
        self._target_dictionary = target_dictionary
        self._source_dictionary = source_dictionary
        self.dataset_files = dataset_files
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "<s>"

        # experiment = Experiment(api_key="fvtehLj4UVIpCNa0qRdSDgkYd",
        #                         project_name="ino-voice-stt",
        #                         workspace="aistis",
        #                         auto_metric_logging=True,
        #                         auto_param_logging=True,
        #                         )
        # experiment.add_tag("unsupervised_pretraining")
        # experiment.set_name("unsupervised_pretraining_0")

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        dataset_files = {}
        if cfg.labels:
            data_files = list(Path(cfg.data).glob('**/*.zip'))
            data_files = list(filter(Path.is_file, data_files))
            if len(data_files) == 0:
                raise Exception("No data file found")
            else:
                random.shuffle(data_files)
                no_of_validation_datasets = max(1, int(len(data_files) * cfg.valid_perc))
                dataset_files["valid"] = data_files[:no_of_validation_datasets]
                dataset_files["train"] = data_files[no_of_validation_datasets:]
            target_dictionary = Dictionary.load(dataset_files["train"])
        else:
            target_dictionary = None

        return cls(cfg, target_dictionary=target_dictionary, dataset_files=dataset_files)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        print("Creating {} dataset from {} batches".format(split,
                                                           len(self.dataset_files[split])))
        self.datasets[split] = FileAudioDataset(
            self.dataset_files[split],
            sample_rate=task_cfg.sample_rate,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.max_sample_size,
            min_length=self.cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
        )

        if task_cfg.labels:
            # label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            # print(label_path)
            # labels = []
            # with open(label_path, "r") as f:
            #     for line in f:
            #         labels.append(line)
            labels = self.get_labels()

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=task_cfg.autoregressive,
            )

    def get_labels(self):
        labels = []
        for k in self.datasets.keys():
            for data_file in self.dataset_files[k]:
                samples = read_manifest_metadata(data_file)
                for sample in samples:
                    transcription = sample.text
                    labels.append(" ".join(list(transcription.replace(" ", "|"))) + " |")
        return labels

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )