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
from .modules import BaseModule
from .linear import Linear, Identity
from .conv import Conv2d
from .dropout import Dropout
from .pooling import MaxPool2d, AdaptiveAvgPool2d, SelectAdaptivePool2d
from .batchnorm import BatchNorm2d
from .activation import ReLU, Softmax
from .loss import EarlyExitClassificationLoss, JoinEarlyExitClassificationLoss
from .residual import ResidualBasicBlock, ResidualBottleneck, make_blocks


__all__ = [
    'BaseModule',
    'Linear', 'Identity',
    'Conv2d',
    'Dropout',
    'MaxPool2d', 'AdaptiveAvgPool2d', 'SelectAdaptivePool2d',
    'BatchNorm2d',
    'ReLU', 'Softmax',
    'EarlyExitClassificationLoss', 'JoinEarlyExitClassificationLoss',
    'ResidualBasicBlock', 'ResidualBottleneck', 'make_blocks'
]
