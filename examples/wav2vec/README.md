# wav2vec 2.0

wav2vec 2.0 learns speech representations on unlabeled data as described in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477).

We also combined wav2vec 2.0 with self-training in [Self-training and Pre-training are Complementary for Speech Recognition (Xu et al., 2020)](https://arxiv.org/abs/2010.11430).


## Training a new model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

### Prepare training data manifest:

Run:

```shell script
$ python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.

$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.
To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv with a
separately pre-processed manifest file.

### Train a wav2vec 2.0 base model:

This configuration was used for the base model trained on the Librispeech dataset in the wav2vec 2.0 paper

Note that this was tested with pytorch 1.4.0 and the input is expected to be single channel, sampled at 16 kHz

```shell script
$ python train.py --distributed-world-size 64 --distributed-port $PORT /manifest/path \
--save-dir /model/path --fp16 --num-workers 6 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 --latent-vars 320 \
--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d
```

Note: you can simulate 64 GPUs by using k GPUs and setting --update-freq 64/k

### Train a wav2vec 2.0 large model:

This configuration was used for the large model trained on the Libri-light dataset in the wav2vec 2.0 paper

```shell script
$ python train.py --distributed-world-size 128 --distributed-port $PORT /manifest/path \
--save-dir /model/path --fp16 --num-workers 6 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 768 --latent-vars 320 \
--latent-groups 2 --latent-temp '(2.0,0.1,0.999995)' --infonce --optimizer adam \
--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 600000 \
--lr 0.0003 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.0 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.03 \
--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --encoder-layers 24 --encoder-embed-dim 1024 \
--encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --num-negatives 100 --cross-sample-negatives 0 \
--max-sample-size 320000 --min-sample-size 32000 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1200000 --max-update 600000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d
```

Note: you can simulate 128 GPUs by using k GPUs and setting --update-freq 128/k

### Fine-tune a pre-trained model with CTC:

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format.
A letter vocabulary can be downloaded [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).
An example [script](libri_labels.py) that generates labels for the Librispeech dataset from the tsv file produced by wav2vec_manifest.py can be used as follows:

```shell script
split=train
$ python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name $split
```

Fine-tuning on 100h of Librispeech with letter targets:
```shell script
valid_subset=dev_other
python train.py --distributed-world-size 24 --distributed-port $PORT /path/to/training_data --save-dir /model/path --fp16 \
--wer-args '("/path/to/lm/4-gram.bin","/path/to/lexicon",2,-1)' \
--post-process letter --valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 \
--max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path /path/to/pretrained/model \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d
```

Note: you can simulate 24 GPUs by using k GPUs and setting --update-freq 24/k

Decoding with a language model during training requires wav2letter [python bindings](https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings).
Alternatively, simply omit the --wer-args flag.

For hyper-parameters to fine-tune other Librispeech splits (10 minutes, 1 hour, etc) please refer to the table in Appendix B in the wav2vec 2.0 paper.
The main changes to make are adjusting --max-update, and then adjusting --warmup-steps, --hold-steps, and --decay steps so that they use 0.1/0.4/0.5 of max-update respectively. You then need to adjust --mask-prob and --mask-channel-prob. This should be set to the mask-length * x where x is the number in the table and mask-length is what you use for --mask-length (10 in this example. Use --mask-channel-length value for --mask-channel-prob).

For example, for 10 hours, we see in the paper that timestep mask prob should be 0.065, so we set --mask-prob to 10* 0.065 = 0.65. channel mask prob is 0.004, so we set it to 64 * 0.004 = 0.256. then we set --max-updates to 20000 and change --warmup-steps to 20000 * 0.1 = 2000, --hold-steps to 8000 and --decay-steps to 10000.

### Evaluating a CTC model:

Evaluating a CTC model with a language model requires wav2letter [python bindings](https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings) to be installed.

Fairseq transformer language model used in the wav2vec 2.0 paper can be obtained from the [wav2letter model repository](https://github.com/facebookresearch/wav2letter/tree/master/recipes/sota/2019).
Be sure to upper-case the language model vocab after downloading it.

Letter dictionary for pre-trained models can be found [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).

Next, run the evaluation command:

```shell script
$subset=dev_other
python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_pretraining \
--nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
--lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter
```

To get raw numbers, use --w2l-decoder viterbi and omit the lexicon. To use the transformer language model, use --w2l-decoder fairseqlm.

## My Instructions

To create wav2vec pretrained model you don't need transcription. Based on wav2vec 2.0 README.md, they suggest 
wavfile lenght 15s to 30s. In my experience, maximum 30s because of GPU memory.


### CREATE PRE-TRAINED MODEL

Put wavs in a specific directory, eg: wav_file. Create a new directory for save the manifest, eg: wav_manifest.
Create a new directory to save results, eg: w2v2_pre_train_model
Run the wav2vec_manifest.py inside fairseq/examples/wav2vec directory with this command (base wav2vec 2.0 README.md):

```
python3 'examples/wav2vec/wav2vec_manifest.py' '/path/to/wav_file' --dest 'path/to/wav_manifest' --ext wav
```
it will create the train.tsv and valid.tsv in your wav_manifest directory.
the start train to make pre-trainned model, by use the command in wav2vec README.md. i choose used the base model on 1 GPU, with command:
```
python3 fairseq/train.py path/to/wav_manifest \
--save-dir path/to/w2v2_pre_train_model --fp16 --num-workers 128 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 --latent-vars 320 \
--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d

```

In trainning to create pre-trainned model, in my experience, it stop automatically when it reaches max result.
After pre-trained model is created, next step is to finetune it with labeled wav.

### FINE TUNING

You must prepare the wav audio. In my assumption 58k hrs labelled data already has all data needed to 
understand speech features and 10 minutes labelled for fine-tuning is enough to map the features to labels.
If you don't have that much unlabelled data, I think the labelled data percentage should be raised.

So, the step to prepare for finetuning are:

Have the label or transcription file in the same folder as the wav file. The transcription file format is:
```
file_name1.wav HI HOW ARE YOU
file_name1.wav THIS IS JUST A SAMPLE OF TRANSCRIPTION FORMAT
file_name3.wav THAT YOU SHOULD BUILD
```
Save the transcription in format folder_name.trans.txt. Illustration: I saved the wav file in folder / directory 
named labelled_wav_file so the name of transcription file is labelled_wav_file.trans.txt and the content of the file 
is in the example above.
You can use either uppercase or lower case letters but not both.

Run the wav2vec_manifest.py again to produce train.tsv and valid.tsv file from labeled audio data.
```
python3 examples/wav2vec/wav2vec_manifest.py /path/to/labeled_wav_file --dest /labbeled_manifest/path --ext wav
```
after that, run twice the file libri_labels.py in fairseq/examples/wav2vec/ directory with command:
``
python3 libri_labels.py /path/to/file/labelled_wav_file/train.tsv --output-dir /path/to/file/labelled_wav_file/ --output-name train
python3 libri_labels.py /path/to/file/labelled_wav_file/valid.tsv --output-dir /path/to/file/labelled_wav_file/ --output-name valid
``
Commands generate train.ltr, train.wrd, valid.ltr and valid.wrd files. There was an error in libri_labels.py 
I already fixed it in the code, so better download the libri_labels.py file again.

If you get an error when running libri_labels.py, replace this script in libri_labels.py:

```
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        root = next(tsv).strip()
        print('root',root)
        for line in tsv:
            line = line.strip()

            dir = os.path.dirname(line)

            if dir not in transcriptions:
                parts = dir.split(os.path.sep)

                trans_path = f"{parts[0]}.trans.txt"

                path = os.path.join(root, dir, trans_path)

                assert os.path.exists(path)
                texts = {}
                with open(path, "r") as trans_f:
                    for tline in trans_f:
                        items = tline.strip().split()
                        texts[items[0]] = " ".join(items[1:])

                transcriptions[dir] = texts
            part = os.path.basename(line).split(".")[0]+'.wav'

            assert part in transcriptions[dir]
            print(transcriptions[dir][part], file=wrd_out)
            print(
                " ".join(list(transcriptions[dir][part].replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()
```
Edit the file train.ltr, train.wrd, valid.ltr and valid.wrd to make sure the file only contains A to Z characters 
(if you use UPPERCASE letter) or a to z character (if you use lower case letter), space and '|' character.

Create a new file named dict.ltr.txt. Open it in a text editor. Use find features in that text editor, I use sublime text.
Then search the character 'A' if you use uppercase letters. The editor will show the number of character 'A'. Note this.
Repeat until all characters are counted. I mean A to Z and '|' character. Then write it base the number of the 
character (ascending base number). example:
```
A 90280
| 78809
N 38268
I 33692
E 30160
K 24305
U 23955
M 20958
T 20551
S 18850
R 18418
D 15972
G 15711
L 13876
B 11768
P 11503
H 10845
Y 9570
O 6758
J 3993
C 2171
W 1118
F 463
V 187
Z 46
X 16
Q 6
```

Save it. remember the file should be named dict.ltr.txt.

Create the lexicon.txt file. I use the train.wrd and valid.wrd to make the lexicon.txt. script i use and write by my self:
```
import os, codecs, re, pandas as pd
a = 'train.wrd'
b = 'valid.wrd'

df1 = pd.read_csv(a, header=None)
df2 = pd.read_csv(b, header=None)

df1.columns = ['raw']
df2.columns = ['raw']

df1 = df1.drop_duplicates('raw',keep='last')
df2 = df2.drop_duplicates('raw',keep='last')

sentence1 = df1['raw'].to_list()
sentence2 = df2['raw'].to_list()
sentence = sentence1 + sentence2

word = []
for x in sentence:
    tmp = x.split(' ')
    for y in tmp:
        if y not in word:
            word.append(y)

lexicon = []
for x in range(len(word)):
    wrd = word[x]
    temp = []
    for y in wrd:
        temp.append(y)
    result = ' '.join(temp) + ' |'
    lexicon.append(wrd + '\t ' + result)

file_to_save = 'lexicon.txt'
f=codecs.open(file_to_save,'a+','utf8')
for x in lexicon:
    f.write(x+'\n')
f.close()
```
it will create file lexicon.txt that have format, something like this:
```
EVERY E V E R Y |
WORD W O R D |
THAT T H A T |
EXISTS E X I S T S |
IN I N |
YOUR Y O U R |
LABEL L A B E L |
OR O R |
TRANSCRIPTION T R A N S C R I P T I O N |
FILE F I L E |
WILL W I L L |
WRITE W R I T E |
DOWN D O W N |
LIKE L I K E |
THIS T H I S |
```

If you use lower case letter in transcription, that make the train.wrd and train.wrd, contains lower case letter., so it become:

```
every e v e r y |
word w o r d |
that t h a t |
etc ...
```

To use kenlm, you need .bin file. Create it by following the instructions in https://youtu.be/NtZipf0BxKg?t=1190.

The command for base fine-tuning:

```
valid_subset=train
python train.py --distributed-world-size 24 --distributed-port $PORT /path/to/training_data --save-dir /model/path --fp16 \
--wer-args '("/path/to/lm/4-gram.bin","/path/to/lexicon",2,-1)' \
--post-process letter --valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 \
--max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path /path/to/pretrained/model \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d
```

the command I used:
```
python3 train.py '/home/bram/Documents/coding/speech/traindata/text_label' \
--save-dir '/home/bram/Documents/coding/speech/traindata/model_finetuning_wav2vec' --fp16 
--wer-args '("/home/bram/Documents/coding/speech/traindata/text_label/lm.bin","/home/bram/Documents/coding/speech/traindata/text_label/lexicon.txt",2,-1)' \
--post-process letter --valid-subset valid --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 128 \
--max-update 400000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc \
--w2v-path '/home/bram/Documents/coding/speech/traindata/w2v2_pre_traned_model/checkpoint_best.pt' \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d
```

The folder /home/bram/Documents/coding/speech/traindata/text_label contains:
```
1. dict.ltr.txt
2. lexicon.txt
3. lm.bin
4. train.tsv
5. train.wrd
6. train.ltr
7. valid.tsv
8. valid.wrd
9. valid.ltr
```
The folder /home/bram/Documents/coding/speech/traindata/w2v2_pre_traned_model/ is contained:
```
1. checkpoint_best.pt
2. checkpoint_last.pt
```
These files came from pre-training process.

To make the pre-trained model have 'args', so it can be read in finetuning, run this script:
```
import torch, argparse, logging, math, os, random, sys, numpy as np
from fairseq import options

# argument is base your command that you use in create pre-trainned model. this is just an example
# last command that i use in run train.py is like this:

 python3 '/content/repo/fairseq/train.py' --distributed-world-size 1 --distributed-port 0 '/content/drive/My Drive/wav_manifest' \
--save-dir '/content/drive/My Drive/wav2vec_v2_pre_train_model' --fp16 --num-workers 128 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 768 --latent-vars 320 \
--latent-groups 2 --latent-temp '(2,0.25,0.999995)' --infonce --optimizer adam \
--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --max-update 600000 \
--lr 0.0004 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.03 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.05 \
--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
--max-sample-size 1500000 --min-sample-size 5000 --dropout 0.05 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1400000 --max-update 600000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --no-epoch-checkpoints

# copy that argument above and convert it to this line above:

argument = [
'/content/repo/fairseq/train.py', '--distributed-world-size', '1', '--distributed-port', '0', '/content/drive/My Drive/wav_manifest',
'--save-dir', '/content/drive/My Drive/wav2vec_v2_pre_train_model', 
'--fp16', '--no-epoch-checkpoints', '--skip-invalid-size-inputs-valid-test', '--infonce','--quantize-targets',
'--num-workers', '128', '--task', 'audio_pretraining', '--criterion', 'wav2vec', '--arch', 'wav2vec2',
'--log-keys', '["prob_perplexity","code_perplexity","temp"]',  '--extractor-mode', 'default',
'--conv-feature-layers', '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2',
'--final-dim', '768', '--latent-vars', '320', '--latent-groups', '2', '--latent-temp', '(2,0.25,0.999995)',  '--optimizer', 'adam',
'--adam-betas', '(0.9,0.98)', '--adam-eps', '1e-06', '--lr-scheduler', 'polynomial_decay', '--max-update', '600000',
'--lr', '0.0004', '--warmup-updates', '32000', '--mask-length', '10', '--mask-prob', '0.65', '--mask-selection', 'static', '--mask-other', '0',
'--encoder-layerdrop', '0.03', '--dropout-input', '0.1', '--dropout-features', '0.1', '--feature-grad-mult', '0.05',
'--loss-weights', '[0.1, 10]', '--conv-pos', '128', '--conv-pos-groups', '16', '--num-negatives', '100', '--cross-sample-negatives', '0',
'--max-sample-size', '1500000', '--min-sample-size', '5000', '--dropout', '0.05', '--attention-dropout', '0.1', '--weight-decay', '0.01',
'--max-tokens', '1400000', '--max-update', '600000',  '--ddp-backend', 'no_c10d', '--encoder-ffn-embed-dim', '4096', '--encoder-attention-heads', '16'
] 
```
