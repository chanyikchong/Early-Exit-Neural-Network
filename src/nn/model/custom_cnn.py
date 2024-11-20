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
import os
from typing import Union, List, Dict, Any

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..modules import *
from .factory import create_model_dir
from .classifier import ClassifierHead

cfgs: Dict[str, Union[dict, List[Union[str, int]]]] = {
    'custom': {
        'layers': [(64, 5, 1, 0), (64, 5, 1, 'same'), (32, 3, 1, 0), 'M', 'M'],
        'last_conv_dim': 512,
    },
    'custom_9increase': {
        'layers': [(64, 3, 1, 0), (64, 3, 1, 'same'), 'M',
                   (128, 3, 1, 'same'), (128, 3, 1, 'same'), 'M',
                   (256, 3, 1, 'same'), (256, 3, 1, 'same'), 'M',
                   (512, 3, 1, 'same'), (512, 3, 1, 'same'), 'M'],
        'last_conv_dim': 512,
    },
    'custom_9plain': {
        'layers': [(64, 3, 1, 0), (64, 3, 1, 'same'), 'M',
                   (128, 3, 1, 'same'), (128, 3, 1, 'same'), 'M',
                   (256, 3, 1, 'same'), (256, 3, 1, 'same'), 'M', 'D'],
        'last_conv_dim': 2304
    }
}

SUPPORT_MODEL = ["custom", "custom_9plain", "custom_9increase"]


class CustomCNN(BaseModel):
    def __init__(
            self,
            cfg: List[Any],
            num_classes: int = 10,
            in_chans: int = 3,
            last_conv_dim: int = 512,
            hidden_features: int = 2048,
            act_layer: nn.Module = ReLU,
            conv_layer: nn.Module = Conv2d,
            norm_layer: nn.Module = None,
            drop_rate: float = 0.,
            **kwargs
    ):
        super(CustomCNN, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_features = hidden_features

        layers: List[nn.Module] = []
        for v in cfg:
            if v == 'M':
                self.append_features(layers, MaxPool2d(kernel_size=2, stride=2), in_chans)
            elif v == 'D':
                self.append_features(layers, Dropout(drop_rate), in_chans)
            elif isinstance(v, tuple):
                self.append_features(layers, conv_layer(in_chans, v[0], v[1], v[2], v[3]), v[0])
                in_chans = v[0]
                self.append_features(layers, act_layer(), in_chans)
                if norm_layer:
                    self.append_features(layers, norm_layer(in_chans), in_chans)

        self.backbone.extend(layers)

        self._initialize_weights()

        self.classifier = ClassifierHead(
            in_features=last_conv_dim,
            num_classes=num_classes,
            hidden_features=hidden_features,
            drop_rate=drop_rate,
        )

        self.classifier_module = ClassifierHead
        self.classifier_config = {
            'in_features': in_chans,
            'hidden_features': hidden_features,
            'num_classes': num_classes,
            'drop_rate': drop_rate,
        }

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)

        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (Conv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm2d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (Linear, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def layer_output(self, x):
        for layer in self.backbone:
            if isinstance(layer, BaseModule):
                x = layer.layer_output(x)
            else:
                x = layer(x)

        x = self.classifier.layer_output(x)
        return x

    def inference(self, x, *args, **kwargs):
        if x.dim() < 4:
            x = x.unsqueeze(dim=0)
        with torch.no_grad():
            return self.forward(x)


def create_custom_cnn(
        name: str = None,
        layers: int = None,
        num_classes: int = 10,
        in_chans: int = 3,
        hidden_feature: int = 2048,
        drop_rate: float = 0.2,
        pretrained: Union[str, bool] = None,
        **kwargs
) -> CustomCNN:
    assert name is not None or layers is not None, "Enter the name or layers of custom cnn"

    assert name.lower() in SUPPORT_MODEL, f"model name is not supported, please chose from the following: {SUPPORT_MODEL}"
    name = name.lower()

    model_cfg = cfgs[name]
    model_cfg_layer = model_cfg['layers']
    last_conv_dim = model_cfg['last_conv_dim']
    if name == 'custom' and layers is not None:
        model_cfg_layer = model_cfg_layer[0:1] + model_cfg_layer[1:2] * layers + model_cfg_layer[2:]

    model = CustomCNN(model_cfg_layer, num_classes=num_classes, in_chans=in_chans, last_conv_dim=last_conv_dim,
                      hidden_feature=hidden_feature, drop_rate=drop_rate, norm_layer=BatchNorm2d, **kwargs)

    if pretrained is not None:
        if isinstance(pretrained, bool) and pretrained:
            pretrained = create_model_dir(name)

        model_path = os.path.join(pretrained, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    return model
