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
from collections import defaultdict
from typing import List

import torch.nn as nn

from ..modules import BaseModule


class PosInitMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = super(PosInitMeta, cls).__call__(*args, **kwargs)
        instance.__pos_init__(**kwargs)
        return instance


class BaseModel(BaseModule, metaclass=PosInitMeta):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()
        self._layer_time_flag = False
        self._layer_memory_flag = False
        self._layer_readable_flag = False
        self.backbone = nn.ModuleList()
        self.feature_info = []

    def __pos_init__(self, **kwargs):
        self.layer_time_flag = kwargs.get('layer_time_flag', False)
        self.layer_memory_flag = kwargs.get('layer_memory_flag', False)

    def append_features(self, layers_list: List, module: nn.Module, next_feature_dim: int, **kwargs):
        layers_list.append(module)
        self.feature_info.append(
            dict(feature_dim=next_feature_dim, module_id=len(layers_list) - 1, module_name=module._get_name(),
                 **kwargs))

    @property
    def layers_execution_time(self):
        module_dict = defaultdict(int)
        time_dict = defaultdict(float)
        for module in self.modules():
            if module != self and isinstance(module, BaseModule):
                module_type = module._get_name()
                module_dict[module_type] += 1
                time_dict[f'{module_type}_{module_dict[module_type]}'] = module.execution_time
        return time_dict

    @property
    def layers_memory(self):
        module_dict = defaultdict(int)
        memory_dict = defaultdict(float)
        for module in self.modules():
            if module != self and isinstance(module, BaseModule):
                module_type = module._get_name()
                module_dict[module_type] += 1
                memory_dict[f'{module_type}_{module_dict[module_type]}'] = module.memory['model']
        return memory_dict

    @property
    def layers_output_memory(self):
        module_dict = defaultdict(int)
        memory_dict = defaultdict(float)
        for module in self.modules():
            if module != self and isinstance(module, BaseModule):
                module_type = module._get_name()
                module_dict[module_type] += 1
                memory_dict[f'{module_type}_{module_dict[module_type]}'] = module.memory['output']
        return memory_dict

    @property
    def backbone_execution_time(self):
        module_dict = defaultdict(int)
        time_dict = defaultdict(float)
        layer_count = 0
        for module in self.backbone.modules():
            if module != self and isinstance(module, BaseModule):
                module_type = module._get_name()
                module_dict[module_type] += 1
                time_dict[f'layer_{layer_count}_{module_type}_{module_dict[module_type]}'] = module.execution_time
                layer_count += 1
        return time_dict

    @property
    def backbone_memory(self):
        module_dict = defaultdict(int)
        memory_dict = defaultdict(float)
        layer_count = 0
        for module in self.backbone.modules():
            if module != self and isinstance(module, BaseModule):
                module_type = module._get_name()
                module_dict[module_type] += 1
                memory_dict[f'layer_{layer_count}_{module_type}_{module_dict[module_type]}'] = module.memory['model']
                layer_count += 1
        return memory_dict

    @property
    def backbone_output_memory(self):
        module_dict = defaultdict(int)
        memory_dict = defaultdict(float)
        layer_count = 0
        for module in self.backbone.modules():
            if module != self and isinstance(module, BaseModule):
                module_type = module._get_name()
                module_dict[module_type] += 1
                memory_dict[f'layer_{layer_count}_{module_type}_{module_dict[module_type]}'] = module.memory['output']
                layer_count += 1
        return memory_dict

    @property
    def layer_time_flag(self):
        return self._layer_time_flag

    @layer_time_flag.setter
    def layer_time_flag(self, value: bool):
        assert isinstance(value, bool), ValueError("Invalid value type")
        self._layer_time_flag = value
        for module in self.modules():
            if module != self and isinstance(module, BaseModule):
                module.time_flag = value

    @property
    def layer_memory_flag(self):
        return self._layer_memory_flag

    @layer_memory_flag.setter
    def layer_memory_flag(self, value: bool):
        assert isinstance(value, bool), ValueError("Invalid value type")
        self._layer_memory_flag = value
        for module in self.modules():
            if module != self and isinstance(module, BaseModule):
                module.memory_flag = value

    @property
    def layer_readable_flag(self):
        return self._layer_readable_flag

    @layer_readable_flag.setter
    def layer_readable_flag(self, value: bool):
        assert isinstance(value, bool), ValueError("Invalid value type")
        self._layer_readable_flag = value
        for module in self.modules():
            if module != self and isinstance(module, BaseModule):
                module.readable_flag = value
