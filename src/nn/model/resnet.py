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
from typing import Union, List, Dict
import re

import torch
import torch.nn as nn

from .base_model import BaseModel
from .classifier import ClassifierHead
from .factory import download_model_config, download_cached_file
from ..modules import *
from ..modules.residual import create_aa

__all__ = ['ResNet', 'create_resnet']

cfgs: Dict[str, Dict[str, List[Union[str, int]]]] = {
    'resnet50': {'channels': [64, 128, 256, 512], 'layers': [3, 4, 6, 3]},
    'resnet101': {'channels': [64, 128, 256, 512], 'layers': [3, 4, 23, 3]},
}

SUPPORT_MODEL = ["resnet50", "resnet50_bn", "resnet50_bn_small",
                 "resnet101", "resnet101_bn", "resnet101_bn_small"]


class ResNetConv1(BaseModule):
    def __init__(
            self,
            in_chans=3,
            inplanes=64,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU,
            deep_stem=False,
            stem_width=64,
            stem_type='',
            small=False,
            **kwargs
    ):
        BaseModule.__init__(self, **kwargs)
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        elif small:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x


class ResNetMaxPool1(BaseModule):
    def __init__(
            self,
            inplanes=64,
            aa_layer=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            replace_stem_pool=False,
            small=False,
            **kwargs
    ):
        BaseModule.__init__(self, **kwargs)
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True),
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            elif small:
                self.maxpool = nn.Identity()
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.maxpool(x)
        return x


class ResNet(BaseModel):
    def __init__(
            self,
            block,
            channels,
            layers,
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            cardinality=1,
            base_width=64,
            stem_width=64,
            stem_type='',
            replace_stem_pool=False,
            block_reduce_first=1,
            down_kernel_size=1,
            avg_down=False,
            act_layer="ReLU",
            norm_layer="BatchNorm2d",
            aa_layer=None,
            drop_rate=0.0,
            drop_path_rate=0.,
            drop_block_rate=0.,
            zero_init_last=True,
            block_args=None,
            small=False,
            hidden_features=False,
            **kwargs
    ):
        super(ResNet, self).__init__(**kwargs)
        time_act = eval(act_layer)
        torch_act = eval(f"nn.{act_layer}")
        if norm_layer:
            time_norm = eval(norm_layer)
            torch_norm = eval(f"nn.{norm_layer}")
        else:
            time_norm = Identity
            torch_norm = nn.Identity

        block_args = block_args or dict()
        kwargs.update(block_args)
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = list()

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        conv1 = ResNetConv1(in_chans, inplanes, norm_layer=torch_norm, act_layer=torch_act, deep_stem=deep_stem,
                            stem_width=stem_width, stem_type=stem_type, small=small, **kwargs)

        bn1 = time_norm(inplanes)
        act1 = time_act(inplace=True)

        maxpool = ResNetMaxPool1(inplanes, aa_layer=aa_layer, act_layer=torch_act, norm_layer=torch_norm,
                                 replace_stem_pool=replace_stem_pool, small=small, **kwargs)

        self.backbone.extend([conv1, bn1, act1, maxpool])

        self.feature_info.append(dict(feature_dim=inplanes, reduction=0, module=conv1._get_name()))
        self.feature_info.append(dict(feature_dim=inplanes, reduction=0, module=bn1._get_name()))
        self.feature_info.append(dict(feature_dim=inplanes, reduction=0, module=act1._get_name()))
        self.feature_info.append(dict(feature_dim=inplanes, reduction=2, module=maxpool._get_name()))

        # Feature Blocks
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=torch_act,
            norm_layer=torch_norm,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )
        for stage in stage_modules:
            self.backbone.extend([*stage[1]])  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)
        for i, d in enumerate(self.feature_info):
            d['module_id'] = i

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        hidden_features = self.num_features // 2 if hidden_features else None

        self.classifier = ClassifierHead(
            in_features=self.num_features,
            hidden_features=hidden_features,
            num_classes=self.num_classes,
            pool_type=global_pool,
            use_softmax=False,
        )

        self.classifier_module = ClassifierHead

        self.classifier_config = {
            'in_features': in_chans,
            'hidden_features': hidden_features,
            'num_classes': num_classes,
            'drop_rate': drop_rate,
            'use_softmax': False,
        }

        self._initialize_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def _initialize_weights(self, zero_init_last=True):
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
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.classifier = ClassifierHead(
            in_features=self.num_features,
            num_classes=self.num_classes,
            pool_type=global_pool,
            use_softmax=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.backbone:
            x = layer(x)

        x = self.classifier(x)
        return x

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


def create_resnet(
        name: str = None,
        num_classes: int = 1000,
        layers: int = None,
        norm: bool = False,
        pretrained: Union[str, bool] = None,
        **kwargs
) -> ResNet:
    """
    Create a ResNet model
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
        match = re.search(r'resnet(\d+)', name)
        layers = match.group(1)

        if "bn" in name:
            norm = True
        else:
            norm = False
        if "small" in name:
            small = True
        else:
            small = False
    block = ResidualBottleneck
    if norm:
        model = ResNet(block=block, channels=cfgs[f'resnet{layers}']['channels'],
                       layers=cfgs[f'resnet{layers}']['layers'], num_classes=num_classes, norm_layer="BatchNorm2d",
                       small=small, **kwargs)
    else:
        model = ResNet(block=block, channels=cfgs[f'resnet{layers}']['channels'],
                       layers=cfgs[f'resnet{layers}']['layers'], num_classes=num_classes, norm_layer=None, small=small,
                       **kwargs)

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
