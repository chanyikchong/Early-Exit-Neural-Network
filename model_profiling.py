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
import shutil
import argparse
import json
from typing import List

from tqdm import tqdm
from PIL import Image

from src.utils import ProfilingHelper, MODE_MAPPING
from src.nn.model import load_model


class ImageLoader:
    def __init__(self, image_folder: str):
        self.image_folder = image_folder
        files = os.listdir(image_folder)
        self._n_sample = sum(1 for file in files if file.endswith('.png'))

    @property
    def n_sample(self):
        return self._n_sample

    def get_image(self, idx):
        image_path = os.path.join(self.image_folder, f"{idx}.png")
        if not os.path.exists(image_path):
            raise ValueError

        image = Image.open(image_path)
        return image


def profile_gate(save_folder, model, image_loader):
    assert hasattr(model, 'gates_layer'), "Invalid model"

    gates_summary = dict()
    path = os.path.abspath(save_folder)
    model.time_flag = True
    for gate, gate_layer in enumerate(model.gates_layer):
        gate_folder = os.path.join(path, f'gate_{gate}')
        profiler = ProfilingHelper(gate_folder, replace=True)
        for i in tqdm(range(image_loader.n_sample)):
            image = image_loader.get_image(i)
            image = model.transform(image)
            model.inference(image, single_gate=gate)
            profiler.execution_time(model.execution_time)
        summary = profiler.summarise(save=False)
        gates_summary[f'execution_time_gate_{gate}_layer_{gate_layer}'] = summary['execution_time']
        shutil.rmtree(gate_folder)
    return gates_summary


def profile_model(model_path, data_path, save_folder, modes: List = None, p_gate: bool = False, human: bool = False):
    """
    :param config: model config file
    :param save_folder: profiling intermedia result save folder
    :param modes:
    :param p_gate:
    :param human: save memory with human-readable format
    :return:
    """
    model = load_model(model_path)
    model.eval()

    image_loader = ImageLoader(data_path)

    # save memory with human-readable format
    if human:
        model.readable_flag = True
        model.layer_readable_flag = True

    if modes:
        profiler = ProfilingHelper(save_folder, replace=True)
    else:
        profiler = ProfilingHelper(save_folder, replace=False)

    # warm up model
    # run inference on first time is slower
    warm_sample = image_loader.get_image(0)
    warm_sample = model.transform(warm_sample)
    model.inference(warm_sample)

    if modes and 0 in modes:
        model.time_flag = True
        for i in tqdm(range(image_loader.n_sample)):
            image = image_loader.get_image(i)
            image = model.transform(image)
            model.inference(image)
            profiler.execution_time(model.execution_time)
        model.time_flag = False
        modes.remove(0)

    if modes and len(modes) > 0:
        model.memory_flag = True
        model.layer_time_flag = True
        model.layer_memory_flag = True
        for i in tqdm(range(image_loader.n_sample)):
            image = image_loader.get_image(i)
            image = model.transform(image)
            model.inference(image)

            for mode in modes:
                if mode % 2 == 0:
                    profiler.layers_profile(MODE_MAPPING[mode], model.__getattribute__(MODE_MAPPING[mode]))

        for mode in modes:
            if mode % 2 == 1:
                if mode == 1:
                    profiler.memory(model.__getattribute__(MODE_MAPPING[mode]))
                else:
                    profiler.layers_profile(MODE_MAPPING[mode], model.__getattribute__(MODE_MAPPING[mode]))

        model.memory_flag = False
        model.layer_time_flag = False
        model.layer_memory_flag = False

    model_summary = profiler.summarise(save=True)

    if p_gate:
        gate_summary = profile_gate(save_folder, model, image_loader)

        for k, v in gate_summary.items():
            model_summary[k] = v

    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, 'profile_summary.json'), 'w', encoding='utf8') as f:
        json.dump(model_summary, f, indent=4)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path', type=str, default='source/models/resnet50_bn_small_ee_cifar_10_end_join')
    parse.add_argument('--data_path', type=str, default='source/images/cifar_10')
    parse.add_argument("--save_folder", type=str, default="source/profiling")
    parse.add_argument("--mode", metavar='N', type=int, nargs='+',
                       help=f"mode of profiling,\n " + "\n ".join([f"{k}: {v}" for k, v in MODE_MAPPING.items()]))
    parse.add_argument('--gate', default=False, action="store_true")
    parse.add_argument("--human", default=False, action="store_true",
                       help="save memory with human-readable format")
    hp = parse.parse_args()

    profile_model(hp.model_path, hp.data_path, hp.save_folder, hp.mode, hp.gate, hp.human)
