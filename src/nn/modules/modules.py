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
import io
import time

import torch
import torch.nn as nn


def bytes_to_readable(num):
    """
    Convert bytes to a readable string representation with units.
    Args:
    - num (int): The number of bytes.

    Returns:
    - str: A string representation of bytes in the format of B, KB, MB, GB, etc.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']:
        if abs(num) < 1024.0:
            return "%3.1f %s" % (num, unit)
        num /= 1024.0
    return "%.1f %s" % (num, 'YB')


def measure_memory(obj, readable=False):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    memory = buffer.getbuffer().nbytes
    buffer.close()
    if readable:
        memory = bytes_to_readable(memory)
    return memory


class BaseModule(nn.Module):
    def __init__(self, **kwargs):
        """
        Base Module for all models in this package
        :param kwargs:
        """
        self._time_flag = kwargs.get('time_flag', False)
        self._memory_flag = kwargs.get('memory_flag', False)
        self._readable_flag = kwargs.get('readable_flag', False)

        self._execution_time = None
        self._self_memory = None
        self._output_memory = None
        self._create_snap_short = None
        self._output_shape = None

        self._init_snap_short()
        super(BaseModule, self).__init__()

    def _init_snap_short(self):
        if self._time_flag and self._memory_flag:
            self._create_snap_short = self._time_memory_snap_short
        elif self.time_flag:
            self._create_snap_short = self._time_snap_short
        elif self.memory_flag:
            self._create_snap_short = self._memory_snap_short
        else:
            self._create_snap_short = self._origin_snap_short

    def _origin_snap_short(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        return wrapper

    def _time_snap_short(self, func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            self._execution_time = time.perf_counter() - start
            return result

        return wrapper

    def _memory_snap_short(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self._self_memory = measure_memory(self.state_dict(), self._readable_flag)
            self._output_memory = measure_memory(result, self._readable_flag)
            return result

        return wrapper

    def _time_memory_snap_short(self, func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            self._execution_time = time.perf_counter() - start

            self._self_memory = measure_memory(self.state_dict(), self._readable_flag)
            self._output_memory = measure_memory(result, self._readable_flag)
            return result
        return wrapper

    def __getattribute__(self, item):
        attr = super().__getattribute__(item)
        if item in ['forward', 'inference'] and callable(attr):
            return self._create_snap_short(attr)
        return attr

    @property
    def execution_time(self):
        return self._execution_time

    @property
    def memory(self):
        return {
            'model': self._self_memory,
            'output': self._output_memory
        }

    @property
    def time_flag(self):
        return self._time_flag

    @time_flag.setter
    def time_flag(self, value: bool):
        assert isinstance(value, bool), ValueError("Invalid value type")
        self._time_flag = value
        self._init_snap_short()

    @property
    def memory_flag(self):
        return self._memory_flag

    @memory_flag.setter
    def memory_flag(self, value: bool):
        assert isinstance(value, bool), ValueError("Invalid value type")
        self._memory_flag = value
        self._init_snap_short()

    @property
    def readable_flag(self):
        return self._readable_flag

    @readable_flag.setter
    def readable_flag(self, value: bool):
        assert isinstance(value, bool), ValueError("Invalid value type")
        self._readable_flag = value
        self._init_snap_short()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        pass

    @property
    def output_shape(self):
        return self._output_shape

    def layer_output(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            self._output_shape = output.shape
        self._output_shape = output.shape
        return output

    @property
    def gflops(self):
        pass
