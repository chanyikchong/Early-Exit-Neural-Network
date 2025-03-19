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
import torch
import torchvision.transforms as T
from timm.data.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
import timm.data.transforms as TimT

from .constants import *


class TransformFactory:
    @classmethod
    def get_transform(cls, trans_type: str, **kwargs) -> (T.Compose, T.Compose):
        train_transform, eval_transform = getattr(cls, trans_type)(**kwargs)
        return train_transform, eval_transform

    @staticmethod
    def vanilla(**kwargs):
        transform = T.Compose([
            T.ToTensor(),
        ])
        return transform, transform

    @staticmethod
    def v1(**kwargs):
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda tensor: tensor / 255)
        ])
        return transform, transform

    @staticmethod
    def v2(**kwargs):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=torch.tensor([0.485, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ])
        return transform, transform

    @staticmethod
    def v3(**kwargs):
        pass

    @staticmethod
    def config(**kwargs):
        """
        Initialize transform with configure
        :param config:
        Keys for initialize transform from configure:\n
        Mandatory key:\n
        input_size (tuple or list): input image's W, H\n
        Optional key:\n
        use_prefetcher (Bool default=False): prefetcher handle tensor conversion and norm\n
        scale (tuple: (lower_bound, upper_bound) default=None): randomly scale the image to the given interval\n
        ratio (tuple: (lower_bound, upper_bound) default=None): randomly taken the ratio of the image\n
        hflip (float default=0.5): horizontal flip\n
        vflip (float default=0): vertical flip\n
        color_jitter (float default=0.4): use color jitter\n
        auto_augment (str default=None): auto augmentation, possible options ['rand', 'augmix']
        check auto_augment_transform in timm.data.auto_augment for more options,\n
        interpolation (str: default='bilinear'): interpolation for resizing, possible options
        ['bilinear', 'nearest', 'nearest_exact', 'bicubic', 'random']\n
        mean (List[float] default={IMAGENET_DEFAULT_MEAN}): scale to mean 0\n
        std (List[float] default={IMAGENET_DEFAULT_STD}: scale to std 1\n
        re_prob (float default=0): random erasing\n
        re_mode (str default='const': random erasing mode\n
        re_count (int default=1): maximum random erasing\n
        re_num_splits (int default=0): number of random erasing split\n
        crop_pct (float default=0): random crop probability \n
        crop_mode (str default='center'): crop mode, possible options ['center', 'squash', 'border']\n

        :param kwargs:
        :return:
        """
        assert kwargs.get('img_size', False), "configure should contain img_size"
        train_key = ['img_size', 'scale', 'ratio', 'hflip', 'vflip', 'color_jitter', 'auto_augment', 'interpolation',
                     'use_prefetcher', 'mean', 'std', 're_prob', 're_mode', 're_count', 're_num_splits']
        eval_key = ['img_size', 'crop_pct', 'crop_mode', 'interpolation', 'use_prefetcher', 'mean', 'std']
        config_for_train = {k: v for k, v in kwargs.items() if k in train_key}
        config_for_eval = {k: v for k, v in kwargs.items() if k in eval_key}
        transform_train = transforms_imagenet_train(**config_for_train)
        transform_eval = transforms_imagenet_eval(**config_for_eval)
        color_transform = T.Lambda(to_rgb)
        transform_train.transforms = [color_transform] + transform_train.transforms
        transform_eval.transforms = [color_transform] + transform_eval.transforms
        return transform_train, transform_eval

    @staticmethod
    def config_help():
        print(
            "Keys for initialize transform from configure:\n"
            "Mandatory key:\n"
            "input_size (tuple or list): input image's W, H\n"
            "Optional key:\n"
            "use_prefetcher (Bool default=False): prefetcher handle tensor conversion and norm\n"
            "scale (tuple: (lower_bound, upper_bound) default=None): randomly scale the image to the given interval\n"
            "ratio (tuple: (lower_bound, upper_bound) default=None): randomly taken the ratio of the image\n"
            "hflip (float default=0.5): horizontal flip\n"
            "vflip (float default=0): vertical flip\n"
            "color_jitter (float default=0.4): use color jitter\n"
            "auto_augment (str default=None): auto augmentation, possible options ['rand', 'augmix'] "
            "check auto_augment_transform in timm.data.auto_augment for more options,\n"
            "interpolation (str: default='bilinear'): interpolation for resizing, possible options "
            "['bilinear', 'nearest', 'nearest_exact', 'bicubic', 'random']\n"
            f"mean (List[float] default={IMAGENET_DEFAULT_MEAN}): scale to mean 0\n"
            f"std (List[float] default={IMAGENET_DEFAULT_STD}: scale to std 1\n"
            "re_prob (float default=0): random erasing\n"
            "re_mode (str default='const': random erasing mode\n"
            "re_count (int default=1): maximum random erasing\n"
            "re_num_splits (int default=0): number of random erasing split\n"
            "crop_pct (float default=0): random crop probability \n"
            "crop_mode (str default='center'): crop mode, possible options ['center', 'squash', 'border']\n"
        )

    @staticmethod
    def eval(transform: T.Compose):
        eval_transforms = []

        for t in transform.transforms:
            if isinstance(t, (T.RandomResizedCrop, T.RandomHorizontalFlip,
                              T.ColorJitter, T.RandomAffine, T.RandomRotation,
                              T.RandomPerspective, T.RandomErasing, T.RandomApply)):
                continue  # Skip augmentation transforms
            elif isinstance(t, TimT.RandomResizedCropAndInterpolation):
                eval_transforms.append(T.Resize(t.size, interpolation=t.interpolation))
            elif isinstance(t, T.Resize):
                eval_transforms.append(t)  # Keep deterministic resizing
            elif isinstance(t, T.RandomResizedCrop):  # Convert RandomResizedCrop to Resize
                eval_transforms.append(T.Resize(t.size))
            else:
                eval_transforms.append(t)  # Keep normalization, ToTensor, etc.

        return T.Compose(eval_transforms)

def to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image
