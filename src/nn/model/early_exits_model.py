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
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from .utils import ModelOutput

from .base_model import BaseModel
from ..modules import BaseModule


def softmax_confidence_func(output):
    return torch.max(output, dim=1)[0]


def normalized_entropy_confidence_func(output):
    return 1 - 1 / torch.log(torch.tensor(output.shape[1])) * torch.sum(-output * torch.log(output), dim=1)


def init_confidence_func(conf_type='max'):
    if conf_type == 'max':
        return softmax_confidence_func
    elif conf_type == 'entropy':
        return normalized_entropy_confidence_func


@dataclass
class EarlyExitsModelOutput(ModelOutput):
    output: torch.Tensor
    exit_layer: Optional[int] = None
    transmit: Optional[bool] = False
    transmit_layer: Optional[int] = None
    confidences: Optional[List[float]] = None
    partition_layers: Optional[List[int]] = None


class EarlyExitsModel(BaseModel):
    """
    Initialized an Early Exits Network from a give Neural Network
    Now only support for classification
    
    Classification model setup:
    Contain an attribute name features which is the backbone of the model
    Contain an attribute name classifier_module which is the class module of the classification head
    Contain an attribute name classifier_config which is a dictionary
    with the input name of the classifier_module as the key and the input value as the value

    The classifier_module should have a key name in_features
    """

    def __init__(self, model: BaseModel, gates_layer: List[int], *args, **kwargs):
        super(EarlyExitsModel, self).__init__(*args, **kwargs)
        self.backbone = model.backbone
        self.feature_info = model.feature_info
        self.classifier_module = model.classifier_module
        self.classifier_config = model.classifier_config

        self.gates_layer = gates_layer
        self.gates = self.generate_exit_gates()
        self.gates[str(len(self.feature_info) - 1)] = model.classifier
        if len(self.feature_info) - 1 not in self.gates_layer:
            self.gates_layer.append(len(self.feature_info) - 1)

        self.confidence_func = init_confidence_func(kwargs.get('confidence_func', 'max'))

    def generate_exit_gates(self):
        gates_dict = nn.ModuleDict()
        for idx, i in enumerate(self.gates_layer):
            assert i <= len(self.feature_info), "gates idx is too large"

            gate_in_feature = self.feature_info[i]['feature_dim']
            classifier_config = self.classifier_config.copy()
            classifier_config['in_features'] = gate_in_feature

            gate_classifier = self.classifier_module(**classifier_config)
            # gate_classifier.gate_layer = i
            gates_dict[str(i)] = gate_classifier
        return gates_dict

    def forward(self, x):
        gate_outputs = list()
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.gates_layer:
                gate_outputs.append(self.gates[str(i)](x))

        gate_outputs = torch.stack(gate_outputs, dim=-1)
        return gate_outputs

    def layer_output(self, x):
        gate_outputs = list()
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, BaseModule):
                x = layer.layer_output(x)
            else:
                x = layer(x)
            if i in self.gates_layer:
                gate_outputs.append(self.gates[str(i)].layer_output(x))

    def _single_exit(self, x, exit_gate, partition_layers: List[int] = None, **kwargs):
        assert exit_gate < len(self.gates_layer), "Invalid exit gate"
        exit_gate = self.gates_layer[exit_gate]

        transmit_layer = None
        if partition_layers:
            transmit_layer = partition_layers.pop(0)

        resume_layer = kwargs.get('resume_layer', None)
        if resume_layer is not None:
            backbone = self.backbone[resume_layer:]
            start_layer_idx = resume_layer
        else:
            backbone = self.backbone
            start_layer_idx = 0

        for i, layer in enumerate(backbone):
            layer_idx = i + start_layer_idx
            # if the current layer is the given transmit layer, stop and return the current tensor
            if transmit_layer is not None and layer_idx == transmit_layer:
                res = EarlyExitsModelOutput(output=x, exit_layer=layer_idx, transmit=True, transmit_layer=layer_idx,
                                            partition_layers=partition_layers)
                return res

            x = layer(x)
            if layer_idx >= exit_gate:
                output = self.gates[str(layer_idx)](x)
                if not self.classifier_config.get('use_softmax', True):
                    output = torch.softmax(output, dim=1)

                confidence = self.confidence_func(output).detach().item()
                output = torch.argmax(output, dim=1).flatten().item()
                res = EarlyExitsModelOutput(output=output, exit_layer=layer_idx, confidences=confidence)
                return res

    def _multi_exit(self, x, multi_gates: List[float], partition_layers: List[list] = None, skip_exit_threshold: float = np.inf,
                    **kwargs):
        assert len(multi_gates) == len(self.gates_layer), "Invalid multi gates"
        exit_gate_threshold = {
            self.gates_layer[i]: threshold for i, threshold in enumerate(multi_gates) if
            0 <= threshold < skip_exit_threshold
        }
        transmit_layer = None
        if partition_layers:
            transmit_layer = partition_layers.pop(0)

        resume_layer = kwargs.get('resume_layer', None)
        if resume_layer is not None:
            backbone = self.backbone[resume_layer:]
            start_layer_idx = resume_layer
        else:
            backbone = self.backbone
            start_layer_idx = 0

        confidences = list()

        for i, layer in enumerate(backbone):
            layer_idx = i + start_layer_idx

            # if the current layer is the given transmit layer, stop and return the current tensor
            if transmit_layer is not None and layer_idx == transmit_layer:
                res = EarlyExitsModelOutput(output=x, exit_layer=layer_idx, transmit=True, transmit_layer=layer_idx,
                                            confidences=confidences, partition_layers=partition_layers)
                return res

            x = layer(x)
            # if the current layer is one of the exit gate, process with the intermediate classifier
            if layer_idx in exit_gate_threshold.keys():
                output = self.gates[str(layer_idx)](x)
                # ensure the output is a probability distribution in order to compare with the threshold
                if not self.classifier_config.get('use_softmax', True):
                    output = torch.softmax(output, dim=1)

                confidence = self.confidence_func(output)
                confidences.append(confidence.detach().item())

                if confidence > exit_gate_threshold[layer_idx]:
                    output = torch.argmax(output, dim=1)[0].item()
                    res = EarlyExitsModelOutput(output=output, exit_layer=layer_idx, confidences=confidences)
                    return res

    def _all_exit(self, x):
        output = self(x)
        if not self.classifier_config.get('use_softmax', True):
            output = torch.softmax(output, dim=1)
        output = torch.max(output, dim=1)
        confidences = output[0].flatten().detach().numpy()
        output = output[1].flatten()
        res = EarlyExitsModelOutput(output=output, exit_layer=len(self.backbone), confidences=confidences)
        return res

    def inference(self, x, single_gate: int = None, multi_gates: List[float] = None, partition_layers: List[int] = None,
                  **kwargs):
        """
        Early Exit Inference can only process one image at a time because the exit gate is different for each image
        :param x: image tensor
        :param single_gate: an integer indicating the exit gate
        :param multi_gates: a list of float indicating the threshold for each exit gate
        :param partition_layers: an integer indicating the layer to pause and transmit the output
        :param return_gate: whether to return the exit gate
        :return:
        """
        if x.dim() < 4:
            x = x.unsqueeze(dim=0)
        with torch.no_grad():
            if single_gate is not None:
                return self._single_exit(x, single_gate, partition_layers, **kwargs)

            elif multi_gates is not None:
                return self._multi_exit(x, multi_gates, partition_layers, **kwargs)

            else:
                return self._all_exit(x)

    @property
    def gate_execution_time(self):
        time_dict = defaultdict(float)
        for gate_layer, gate in self.gates.items():
            gate_id = self.gates_layer.index(int(gate_layer))
            module_dict = defaultdict(int)
            for module in gate.modules():
                if module != self and isinstance(module, BaseModule):
                    module_type = module._get_name()
                    module_dict[module_type] += 1
                    time_dict[
                        f'gate_{gate_id}_layer_{gate_layer}_{module_type}_{module_dict[module_type]}'] = module.execution_time
        return time_dict

    @property
    def gate_memory(self):
        memory_dict = defaultdict(float)
        for gate_layer, gate in self.gates.items():
            gate_id = self.gates_layer.index(int(gate_layer))
            module_dict = defaultdict(int)
            for module in gate.modules():
                if module != self and isinstance(module, BaseModule):
                    module_type = module._get_name()
                    module_dict[module_type] += 1
                    memory_dict[f'gate_{gate_id}_layer_{gate_layer}_{module_type}_{module_dict[module_type]}'] = \
                        module.memory['model']
        return memory_dict

    @property
    def gate_output_memory(self):
        memory_dict = defaultdict(float)
        for gate_layer, gate in self.gates.items():
            gate_id = self.gates_layer.index(int(gate_layer))
            module_dict = defaultdict(int)
            for module in gate.modules():
                if module != self and isinstance(module, BaseModule):
                    module_type = module._get_name()
                    module_dict[module_type] += 1
                    memory_dict[f'gate_{gate_id}_layer_{gate_layer}_{module_type}_{module_dict[module_type]}'] = \
                        module.memory['output']
        return memory_dict


def create_early_exit_network(model, gates_layers, pretrained: str = None, **kwargs):
    kwargs['model'] = model
    kwargs['gates_layer'] = gates_layers
    model = EarlyExitsModel(**kwargs)
    if pretrained and isinstance(pretrained, str):
        model.load_state_dict(torch.load(f"{pretrained}/model.pth", map_location='cpu'), strict=False)
    return model
