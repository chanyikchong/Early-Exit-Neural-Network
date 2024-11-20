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
from typing import Tuple, Union
import torch.nn as nn

from .modules import BaseModule


__all__ = ['MaxPool2d', 'AdaptiveAvgPool2d', 'SelectAdaptivePool2d']

_int_tuple_2_t = Union[int, Tuple[int, int]]


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


class MaxPool2d(nn.MaxPool2d, BaseModule):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, **kwargs):
        BaseModule.__init__(self, **kwargs)
        nn.MaxPool2d.__init__(self, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, BaseModule):
    def __init__(self, output_size, **kwargs):
        BaseModule.__init__(self, **kwargs)
        nn.AdaptiveAvgPool2d.__init__(self, output_size)


class SelectAdaptivePool2d(BaseModule):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHW',
            **kwargs
    ):
        super(SelectAdaptivePool2d, self).__init__(**kwargs)
        assert input_fmt in ('NCHW', 'NHWC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        else:
            assert input_fmt == 'NCHW'
            if pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            else:
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'pool_type=' + self.pool_type \
            + ', flatten=' + str(self.flatten) + ')'
