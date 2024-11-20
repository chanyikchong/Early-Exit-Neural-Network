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
from typing import Union

from numpy import ndarray
import torch
from torch import Tensor
import torch.nn as nn

from .modules import BaseModule


class EarlyExitClassificationLoss(BaseModule):
    def __init__(
            self, loss_fn: nn.Module,
            num_gates: int,
            gate_weights: Union[list, ndarray, Tensor] = None,
            **kwargs
    ):
        BaseModule.__init__(self, **kwargs)
        self.loss_fn = loss_fn
        if self.loss_fn.reduction != 'none':
            self.loss_fn.reduction = 'none'
        self.num_gates = num_gates
        self.gate_weights = torch.tensor(gate_weights) if gate_weights else torch.ones(self.num_gates)

    @staticmethod
    def process_output(output: Tensor):
        if isinstance(output, Tensor) and output.dim() < 3:
            output = output.unsqueeze(dim=-1)
        elif isinstance(output, dict):
            output = list(output.values())
            output = torch.stack(output, dim=-1)
        elif isinstance(output, list):
            output = torch.stack(output, dim=-1)
        return output

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        output = self.process_output(output)
        if target.dim() == 1:
            target = target.unsqueeze(dim=1)
        target = target.repeat((1, self.num_gates))
        loss = self.loss_fn(output, target)
        loss = loss.mean(dim=0)
        loss = torch.matmul(self.gate_weights.to(output.device), loss)
        return loss


class JoinEarlyExitClassificationLoss(EarlyExitClassificationLoss):
    def __init__(
            self,
            loss_fn: nn.Module,
            num_gates: int,
            start_weight: float = 0.01,
            n_incremental: int = 10,
            **kwargs
    ):
        super(JoinEarlyExitClassificationLoss, self).__init__(loss_fn, num_gates, **kwargs)
        self.gate_max_weights = torch.linspace(0, 1, num_gates + 1)[1:]
        self.gate_weights = torch.minimum(torch.ones(num_gates) * start_weight, self.gate_max_weights)
        self.increment_step = (self.gate_max_weights - self.gate_weights) / n_incremental

    def step(self):
        self.gate_weights = torch.minimum(self.gate_max_weights, self.gate_weights + self.increment_step)


class SelectGateEarlyExitClassificationLoss(nn.Module):
    def __init__(
            self,
            loss_fn: nn.Module,
            gate_id: int
    ):
        super(SelectGateEarlyExitClassificationLoss, self).__init__()
        self.loss_fn = loss_fn
        if self.loss_fn.reduction == 'none':
            self.loss_fn.reduction = 'mean'
        self.gate_id = gate_id

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        if output.dim() > 2:
            output = output[:, :, self.gate_id]
        loss = self.loss_fn(output, target)
        return loss
