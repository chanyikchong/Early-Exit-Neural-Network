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
import json
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import torch

from ..nn.model import generate_model
from ..data.datasets import load_dataset
from ..data.transform import create_transform


class LayerExecutionTimeExp:
    def __init__(self, model_name, dataset_name, save_folder, split='test', streaming=True):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder

        self.model = generate_model(model_name, pretrained=True)
        self.model.time_flag = True
        self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        self.transform = create_transform(self.model, color_type='RGB')

    def run(self, device='cpu', num_sample=None, save=True):
        """
        Run experiment
        :param device: specify the device where the data and model are run from
        :param num_sample: specify the number of sample. If not specify, run the complete dataset
        :return:
        a dictionary contain the experiment result in the following form
        {'acc': accuracy, 'execution_time': execution_time_list}
        """
        self.model.eval()
        acc = 0
        count = 0
        execution_time_dict = defaultdict(list)

        for i, sample in enumerate(tqdm(self.dataset, total=num_sample)):
            img = sample['image']
            label = sample['label']
            img = self.transform(img).unsqueeze(0)
            img = img.to(device)
            output = self.model(img)
            y_hat = torch.argmax(output, dim=1)[0]

            if y_hat == label:
                acc += 1
            count += 1

            execution_time = self.model.layers_execution_time()
            for k, v in execution_time.items():
                execution_time_dict[k].append(v)

            if num_sample is not None and i >= num_sample:
                break

        acc = acc / count
        result = {'acc': acc, 'execution_time': execution_time_dict}
        if save:
            self.save_result(result)
        return result

    def save_result(self, result):
        with open(os.path.join(self.save_folder, 'result.json'), 'w', encoding='utf8') as f:
            json.dump(result, f)

        execution_time = result['execution_time']
        df = pd.DataFrame(execution_time)
        df.to_csv(os.path.join(self.save_folder, 'execution_time.csv'), index=False)
