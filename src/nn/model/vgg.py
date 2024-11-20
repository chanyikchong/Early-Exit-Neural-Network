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
import os.path
from typing import Union, List, Dict, Any, cast
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .classifier import ClassifierHead
from ..modules import *
from .factory import download_model_config, download_cached_file

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

SUPPORT_MODEL = ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]


class ConvMlp(nn.Module):

    def __init__(
            self,
            in_features=512,
            out_features=4096,
            kernel_size=7,
            mlp_ratio=1.0,
            drop_rate: float = 0.2,
            act_layer: nn.Module = None,
            conv_layer: nn.Module = None,
    ):
        super(ConvMlp, self).__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(True)
        self.drop = Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(True)

    def forward(self, x):
        if x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            # keep the input size >= 7x7
            output_size = (max(self.input_kernel_size, x.shape[-2]), max(self.input_kernel_size, x.shape[-1]))
            x = F.adaptive_avg_pool2d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class ClassifierModule(nn.Module):
    def __init__(
            self,
            in_features=512,
            hidden_features=4096,
            num_classes=1000,
            kernel_size=7,
            mlp_ratio=1.0,
            drop_rate: float = 0.2,
            act_layer: nn.Module = None,
            conv_layer: nn.Module = None,
            pool_type: str = 'avg',
            **kwargs
    ):
        super(ClassifierModule, self).__init__()
        self.pre_logits = ConvMlp(
            in_features,
            hidden_features,
            kernel_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            act_layer=act_layer,
            conv_layer=conv_layer,
        )
        self.head = ClassifierHead(
            hidden_features,
            num_classes,
            pool_type=pool_type,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)


class VGG(BaseModel):
    def __init__(
            self,
            cfg: List[Any],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            mlp_ratio: float = 1.0,
            act_layer: nn.Module = ReLU,
            conv_layer: nn.Module = Conv2d,
            norm_layer: nn.Module = None,
            global_pool: str = 'avg',
            drop_rate: float = 0.2,
            **kwargs
    ) -> None:
        super(VGG, self).__init__(**kwargs)
        assert output_stride == 32

        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.use_norm = norm_layer is not None

        prev_chs = in_chans
        net_stride = 1
        pool_layer = MaxPool2d
        layers: List[nn.Module] = []

        for v in cfg:
            if v == 'M':
                self.append_features(layers, pool_layer(kernel_size=2, stride=2), prev_chs)
            elif v == 'D':
                self.append_features(layers, Dropout(drop_rate), prev_chs)
            else:
                v = cast(int, v)
                self.append_features(layers, conv_layer(prev_chs, v, kernel_size=3, padding=1), v)
                prev_chs = v
                if norm_layer:
                    self.append_features(layers, norm_layer(prev_chs), prev_chs)
                self.append_features(layers, act_layer(inplace=True), prev_chs)

        self.backbone.extend(layers)

        self.classifier_kernel_size = 7

        self.classifier = ClassifierModule(
            in_features=prev_chs,
            hidden_features=self.num_features,
            num_classes=num_classes,
            kernel_size=self.classifier_kernel_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            act_layer=act_layer,
            conv_layer=conv_layer,
            pool_type=global_pool
        )

        self._initialize_weights()

        self.classifier_module = ClassifierModule
        self.classifier_config = {
            "in_features": 512,
            "hidden_features": self.num_features,
            "num_classes": num_classes,
            "kernel_size": self.classifier_kernel_size,
            "mlp_ratio": mlp_ratio,
            "drop_rate": drop_rate,
            "act_layer": act_layer,
            "conv_layer": conv_layer,
            "pool_type": global_pool
        }

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        # this treats BN layers as separate groups for bn variants, a lot of effort to fix that
        return dict(stem=r'^features\.0', blocks=r'^features\.(\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.classifier.head = ClassifierHead(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def inference(self, x, *args, **kwargs):
        if x.dim() < 4:
            x = x.unsqueeze(dim=0)
        with torch.no_grad():
            return self.forward(x)

def create_vgg(
        name: str = None,
        num_classes: int = 1000,
        layers: int = None,
        norm: bool = False,
        pretrained: Union[str, bool] = None,
        **kwargs
) -> VGG:
    """
    Create a VGG model
    :param name:
    :param num_classes:
    :param layers:
    :param norm:
    :param pretrained:
    :return:
    """
    assert name is not None or layers is not None, "Either the name of the model or number the layers shoud be provided"

    if name is not None:
        assert name.lower() in SUPPORT_MODEL, f"model name is not supported, please chose from the following: {SUPPORT_MODEL}"
        name = name.lower()
        match = re.search(r'vgg(\d+)', name)
        layers = match.group(1)

        if "bn" in name:
            norm = True
        else:
            norm = False

    if norm:
        model = VGG(cfgs[f'vgg{layers}'], num_classes=num_classes, norm_layer=BatchNorm2d)
    else:
        model = VGG(cfgs[f'vgg{layers}'], num_classes=num_classes)

    if pretrained:
        if isinstance(pretrained, str):
            model_path = os.path.join(pretrained, "model.pth")
            config_path = os.path.join(pretrained, "config.json")

        elif isinstance(pretrained, bool) and pretrained:
            model_name = f"vgg{layers}_bn" if norm else f"vgg{layers}"
            config_path = download_model_config(
                f"https://huggingface.co/timm/{model_name}.tv_in1k/resolve/main/config.json", model_name)
            model_path = download_cached_file(
                f"https://huggingface.co/timm/{model_name}.tv_in1k/resolve/main/pytorch_model.bin", model_name)
        else:
            return model

        model.load_state_dict(torch.load(model_path), strict=False)

    return model
