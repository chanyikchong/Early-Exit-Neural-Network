"""
Copyright (c) [2024], [Yichong Chen]
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree or at the following link:
https://opensource.org/licenses/BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import json
import os
from typing import Union, List

import torch

from .resnet import create_resnet, ResNet
from .vgg import create_vgg, VGG
from .custom_cnn import create_custom_cnn, CustomCNN
from .early_exits_model import create_early_exit_network, EarlyExitsModel
from ...data.transform import TransformFactory

__all__ = [
    'generate_model', 'load_model',
    'create_custom_cnn', 'CustomCNN',
    'create_vgg', 'VGG',
    'create_early_exit_network', 'EarlyExitsModel'
]

SUPPORT_MODELS = [
    "resnet50", "resnet50_bn", "resnet50_bn_small", "resnet101", "resnet101_bn",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    "custom", "custom_9plain", "custom_9increase",
]

SUPPORT_MODELS_EARLY_EXIST = [f"{model}_ee" for model in SUPPORT_MODELS]

SUPPORT_MODELS += SUPPORT_MODELS_EARLY_EXIST


def generate_model(model_name: str, pretrained: Union[bool, str] = None, early_exits_gates: List = None, **kwargs):
    f"""
    Generate Model from model name
    Current support model names:
    {SUPPORT_MODELS}
    
    [CAUTIONS] if you want to load a trained early exit network, please use load_model method. Early exit network 
    generated from this method does not load the gates parameters
    
    :param model_name: The name of the model
    :param pretrained: load pretrained model
    :param early_exits_gates: Early exit gates
    :return: PyTorch model
    """
    assert model_name.lower() in SUPPORT_MODELS, f"{model_name} not supported"

    if "vgg" in model_name.lower():
        model = create_vgg(model_name.replace('_ee', ''), pretrained=pretrained, **kwargs)

    elif "custom" in model_name.lower():
        kwargs['layers'] = kwargs.get('layers', None)
        model = create_custom_cnn(model_name.replace('_ee', ''), pretrained=pretrained, **kwargs)
    elif "resnet" in model_name.lower():
        model = create_resnet(model_name.replace('_ee', ''), pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"{model_name} not supported")

    if model_name.split("_")[-1] == 'ee':
        model = create_early_exit_network(model, early_exits_gates, **kwargs)

    return model


def load_model(model_path: str):
    with open(os.path.join(model_path, "config.json"), 'r', encoding='utf8') as f:
        config = json.load(f)

    if "early_exits_gates" in config.keys() and 'gates_layer' not in config.keys():
        config['gates_layer'] = config['early_exits_gates']
    model = generate_model(**config)
    model.pretrained_cfg = config['pretrained_cfg']

    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location="cpu"))
    transform_type = config['pretrained_cfg'].get('transform', None)
    if transform_type:
        transform_detail = config['pretrained_cfg'].get('transform_config', dict())
        _, model.transform = TransformFactory.get_transform(transform_type, **transform_detail)
    else:
        _, model.transform = TransformFactory.get_transform("vanilla")
    return model
