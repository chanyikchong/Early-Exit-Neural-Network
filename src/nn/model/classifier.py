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
from typing import Optional

import torch.nn as nn

from ..modules import *


def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv, \
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, hidden_features=None, use_conv=False, use_softmax=True):
    if num_classes <= 0:
        fc = Identity()  # pass-through (no classifier)
    elif use_conv:
        if hidden_features:
            fc = nn.Sequential(
                Conv2d(num_features, hidden_features, 1, bias=True),
                ReLU(),
                Conv2d(hidden_features, num_classes, 1, bias=True),
                Softmax(dim=-1) if use_softmax else Identity(),
            )
        else:
            fc = nn.Sequential(
                Conv2d(num_features, num_classes, 1, bias=True),
                Softmax(dim=-1) if use_softmax else Identity(),
            )
    else:
        if hidden_features:
            fc = nn.Sequential(
                Linear(num_features, hidden_features, bias=True),
                ReLU(),
                Linear(hidden_features, num_classes, bias=True),
                Softmax(dim=1) if use_softmax else Identity(),
            )
        else:
            fc = nn.Sequential(
                Linear(num_features, num_classes, bias=True),
                Softmax(dim=1) if use_softmax else Identity(),
            )
    return fc


def create_classifier(
        num_features: int,
        num_classes: int,
        hidden_features: int = None,
        pool_type: str = 'avg',
        use_conv: bool = False,
        use_softmax: bool = True,
        input_fmt: str = 'NCHW',
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        hidden_features=hidden_features,
        use_conv=use_conv,
        use_softmax=use_softmax
    )
    if drop_rate is not None:
        dropout = Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            hidden_features: int = None,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False,
            use_softmax: bool = True,
            input_fmt: str = 'NCHW',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_conv = use_conv
        self.use_softmax = use_softmax
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            hidden_features=hidden_features,
            pool_type=pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
            use_softmax=use_softmax
        )
        self.global_pool = global_pool
        self.drop = Dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes, pool_type=None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
                use_softmax=self.use_softmax
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)

    def layer_output(self, x, pre_logits: bool = False):
        x = self.global_pool.layer_output(x)
        x = self.drop.layer_output(x)
        if pre_logits:
            return self.flatten(x)
        for layer in self.fc:
            x = layer.layer_output(x)
        return self.flatten(x)
