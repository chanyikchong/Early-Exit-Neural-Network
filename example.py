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
from os.path import split


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

    import time

    eenn = True
    is_imagenet = True

    if is_imagenet:
        if eenn:
            model_path = "source/models/resnet101_bn_ee_imagenet_1k_end_join"  # Early exit model
        else:
            model_path = "source/models/resnet101_bn_backbone"  # Regular model
        dataset = 'imagenet_1k'
        splits = 'validation'
        img = 'image'
        multi_gate = [0.91070014,
                      0.9536827,
                      0.83667094,
                      0.87508076,
                      0.9126322,
                      0.8780481,
                      0.8828466,
                      0.8660141,
                      0.86336315,
                      0.8740528,
                      0.89549655,
                      0.89989233,
                      0.90021265,
                      0.8881942,
                      0.874336,
                      0.8798743,
                      0.901095,
                      0.88530785,
                      0.9016632,
                      0.8882216,
                      0.8954079,
                      0.9099823,
                      0.9152535,
                      0.87924534,
                      0.8616251,
                      0.8693555,
                      0.8826693,
                      0.8942911,
                      0.9056602,
                      0.827167,
                      0.8389424,
                      0.7379005,
                      0]

    else:
        if eenn:
            model_path = "source/models/resnet50_bn_small_ee_cifar_10_end_join"  # Early exit model
        else:
            model_path = "source/models/resnet50_bn_small_backbone"  # Regular model
        dataset = 'cifar_10'
        splits = 'test'
        img = 'img'
        multi_gate = [0.57032263,
                      0.7698907,
                      0.8529816,
                      0.8673227,
                      0.8883128,
                      0.8604843,
                      0.7554105,
                      0.64174026,
                      0]

    model = load_model(model_path)

    data = load_dataset(dataset, split=splits, streaming=False, cache_dir="source/datasets")
    data.set_transform(partial(set_transform, transform=model.transform))
    dataloader = DataLoader(data, batch_size=1)
    eval_metric = EpochEvalMetric()
    model.eval()

    time_sum = 0
    for batch in tqdm(dataloader):
        x = batch[img]
        y = batch['label']

        start = time.perf_counter()
        if eenn:
            y_hat = model.inference(x, multi_gates=multi_gate)
            y_hat = y_hat.output
        else:
            y_hat = model(x)
        time_sum += time.perf_counter() - start
        eval_metric.update(x.shape[0], torch.tensor(0), y_hat, y)

    print(eval_metric.eval_info())
    print("Average inference time:", time_sum / len(dataloader))


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

        threshold = np.random.uniform(0, 1, len(model.gates_layer) - 1).tolist()
        threshold.append(0)  # The threshold for the last layer is always 0
        output = model.inference(image, multi_gates=threshold)
        print("Multiple gates inference:", output)


if __name__ == '__main__':
    example_inference_on_dataset()
    # example_inference_on_single_image()
