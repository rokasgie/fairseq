from fairseq.checkpoint_utils import load_checkpoint_to_cpu
import torch

pretrained = "/home/jupyter/ino-voice/models/pretrained/xlsr/xlsr_53_56k_adjusted.pt"
current = "/home/jupyter/ino-voice/models/finetuned/xlsr-lt/checkpoint_best.pt"
adjusted = "/home/jupyter/ino-voice/models/finetuned/xlsr-lt/checkpoint_best_adjusted.pt"

state = load_checkpoint_to_cpu(current, {})
pretrained_state = load_checkpoint_to_cpu(pretrained, {})

state["cfg"].model.w2v_args = pretrained_state["cfg"]
state["cfg"].model.w2v_path = ""
# tmp_state["cfg"].model.w2v_args["task"] = tmp_state["cfg"].task
# tmp_state["cfg"].model.w2v_args["model"] = tmp_state["cfg"].model

torch.save(state, adjusted)
