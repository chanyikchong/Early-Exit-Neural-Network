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
import os


# Examples for using the EENN package

def example_inference_on_dataset():
    from functools import partial
    from tqdm import tqdm

    import torch
    from torch.utils.data import DataLoader

    from src.nn.model import load_model
    from src.data.datasets import load_dataset
    from src.data.transform import set_transform
    from src.experiments.train_image_classification import EpochEvalMetric

    model_path = "source/models/resnet50_bn_small_backbone"  # Regular model
    # model_path = "source/models/resnet50_bn_small_ee_cifar_10_end_join"  # Early exit model

    model = load_model(model_path)

    dataset = 'cifar_10'
    data = load_dataset(dataset, split="test", streaming=False, cache_dir="source/datasets")
    data.set_transform(partial(set_transform, transform=model.transform))
    dataloader = DataLoader(data, batch_size=128)
    eval_metric = EpochEvalMetric()
    model.eval()
    for batch in tqdm(dataloader):
        x = batch['img']
        y = batch['label']
        y_hat = model(x)
        eval_metric.update(x.shape[0], torch.tensor(0), y_hat, y)

    print(eval_metric.eval_info())


def example_inference_on_single_image():
    from PIL import Image
    import numpy as np

    from src.nn.model import load_model
    from src.nn.model import EarlyExitsModel

    model_path = "source/models/resnet50_bn_small_backbone"  # Regular model
    # model_path = "source/models/resnet50_bn_small_ee_cifar_10_end_join"  # Early exit model
    model = load_model(model_path)

    dataset = 'cifar_10'
    image_path = f'source/images/{dataset}/0.png'
    image = Image.open(image_path)
    image = model.transform(image)

    # No gate inference
    output = model.inference(image)
    print("No gate inference:", output)

    if isinstance(model, EarlyExitsModel):
        # Single gate inference
        output = model.inference(image, single_gate=4)
        print("Single gate inference:", output)

        # Multiple gates inference

        threshold = np.random.uniform(0, 1, len(model.gates_layer)-1).tolist()
        threshold.append(0)  # The threshold for the last layer is always 0
        output = model.inference(image, multi_gates=threshold)
        print("Multiple gates inference:", output)


if __name__ == '__main__':
    example_inference_on_dataset()
    example_inference_on_single_image()
