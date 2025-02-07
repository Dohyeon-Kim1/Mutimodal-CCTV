import os
import torch

from models.vast.vast import VAST
from models.vast.build_args import get_args

def build_vast():
    args = get_args()
    model = VAST(args.model_cfg)

    sd = torch.load("models/vast/pretrained_weights/vast.pth")
    model.load_state_dict(sd)
    return model