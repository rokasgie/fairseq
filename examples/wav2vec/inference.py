import argparse
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from tqdm import tqdm
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pathlib import Path

from fairseq.data import Dictionary
from fairseq.checkpoint_utils import load_model_ensemble

parser = argparse.ArgumentParser("Wav2Vec2 Inference")
parser.add_argument("--data-dir", type=str, required=True, help="Directory where wavs are stored")
parser.add_argument("--model-path", type=str, required=True, help="Path to finetuned wav2vec2 model")
parser.add_argument("--dict-path", type=str, required=True, help="Path to dictionary file")
parser.add_argument("--sample-rate", type=int, default=22050, required=False, help="Sample rate")
parser.add_argument("--batch-size", type=int, default=1, required=False, help="Batch size")
args = parser.parse_args()


def postprocess(feats, curr_sample_rate, normalize=True):
    if args.sample_rate != curr_sample_rate:
        feats = Resample(curr_sample_rate, args.sample_rate)(feats)

    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats


def get_item(audio_file: Path, normalize):
    if audio_file.suffix == ".wav":
        waveform, sample_rate = read_wav(audio_file)
    elif audio_file.suffix == ".mp3":
        waveform, sample_rate = read_mp3(audio_file)
    else:
        raise ValueError(f"File extension for {audio_file} was not recognized.")

    feats = torch.from_numpy(waveform).float()
    return postprocess(feats, sample_rate, normalize)


def read_wav(file):
    waveform, sample_rate = sf.read(file, dtype="float32")
    return waveform, sample_rate


def read_mp3(file, normalized=False):
    """MP3 to numpy array"""
    audio = AudioSegment.from_mp3(file)
    audio = audio.set_channels(1)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if normalized:
        return y / 2 ** 15, audio.frame_rate, audio.duration_seconds
    else:
        return y, audio.frame_rate, audio.duration_seconds


class Decoder:
    def __init__(self, dict_file):
        self.dict = Dictionary.load(dict_file)

    def decode(self, tensor):
        output = self.dict.string(tensor)
        return output


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    decoder: Decoder = Decoder(args.dict_path)
    model, cfg = load_model_ensemble([args.model_path],
                                     arg_overrides={"data": '/home/rokas/manifests/full'})
    model = model[0].eval().cuda()

    files = list(Path(args.data_dir).glob('*.wav')) + list(Path(args.data_dir).glob('*.mp3'))
    for i, wav in enumerate(files):
        try:
            with open(str(wav)[:-3] + "txt", "r") as file:
                text = file.read().strip()
        except:
            text = str(wav)[:-4].split("__")[-1]

        input_sample = get_item(wav, cfg["task"]["normalize"]).unsqueeze(0)
        logits = model(source=input_sample.cuda(), padding_mask=None)["encoder_out"].cpu()

        predicted_ids = torch.argmax(logits[:, 0], axis=-1)
        prediction = decoder.decode(predicted_ids)
        prediction = prediction.replace(' ', '').replace('|', ' ').replace('  ', " ")
        print("{}: {}\n{}: {}\n".format(i, text, i, prediction))

