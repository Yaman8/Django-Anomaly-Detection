from __future__ import division
import torch
from torch import nn
from .resnext import resnet101
import pdb

def generate_model():
    model = resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            input_channels=3,
            output_layers=[])
    

    model = model
    model = nn.DataParallel(model)

    return model

