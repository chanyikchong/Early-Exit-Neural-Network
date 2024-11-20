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
import argparse
import json
import os.path
import traceback

import numpy as np
from datetime import datetime
from functools import partial
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.experiments.train_image_classification import load_train_mode
from src.nn.model import generate_model, load_model
from src.data.datasets import load_dataset
from src.data.transform import TransformFactory, set_transform


def set_optimizer(optim_mode: str):
    optim_mode = optim_mode.lower()
    if optim_mode == 'sgd':
        optimizer = SGD
    elif optim_mode == 'adam':
        optimizer = Adam
    else:
        raise ValueError("unsupport optimizer")
    return optimizer


def set_scheduler(scheduler_mode: str):
    schedule_mode = scheduler_mode.lower()
    if schedule_mode == 'reducelronplateau':
        scheduler = ReduceLROnPlateau
    else:
        raise ValueError("unsupport scheduler")
    return scheduler


def main(hp):
    if hp.seed:
        torch.manual_seed(hp.seed)
        torch.cuda.manual_seed(hp.seed)

    model_config = dict()

    # argument settings
    dataset_cache_dir = 'source/datasets'
    device = 'cuda:0' if hp.cuda and torch.cuda.is_available() else 'cpu'

    dataset = hp.dataset
    num_classes = hp.num_classes
    dataset_name = f"{dataset}_{num_classes}" if num_classes < 1000 else f"{dataset}_{num_classes // 1000}k"
    input_size = [3, 32, 32] if dataset == 'cifar' else [3, 224, 224]

    model_config['dataset'] = dataset_name
    model_config['num_classes'] = num_classes
    model_config['input_size'] = input_size

    model_name = hp.model

    # train early exit network
    if hp.ee:
        model_name = f"{model_name}_ee"
        model_config['early_exits_gates'] = hp.ee

    model_config["confidence_func"] = hp.confidence_score
    model_config['model_name'] = model_name

    batch_size = hp.batch
    lr = hp.lr
    weight_decay = hp.decay
    momentum = hp.momentum
    epochs = hp.epochs

    pretrained = None
    if hp.pretrained is not None:
        if hp.pretrained.lower() == 'y':
            pretrained = True
        elif hp.pretrained.lower() == 'n':
            pretrained = False
        else:
            pretrained = hp.pretrained

    # create dataset
    print(dataset_cache_dir)
    train_dataset = load_dataset(dataset_name, split="train", streaming=False, cache_dir=dataset_cache_dir)
    eval_dataset = load_dataset(dataset_name, split=hp.test_split, streaming=False, cache_dir=dataset_cache_dir)

    # create transform
    if pretrained and os.path.exists(os.path.join(hp.pretrained, "config.json")):
        # load pretrained model config
        with open(os.path.join(hp.pretrained, "config.json"), 'r', encoding='utf8') as f:
            pretrained_config = json.load(f)
        pretrained_cfg = pretrained_config['pretrained_cfg']
        transform_type = pretrained_cfg['transform']
        transform_detail = pretrained_cfg.get('transform_config', dict())
        transform_train, transform_eval = TransformFactory.get_transform(transform_type, **transform_detail)
    else:
        pretrained_cfg = dict()
        transform_conf = dict()
        if hp.transform == 'config':
            assert hp.transform_conf is not None, "transform configure file path is not provided"
            with open(hp.transform_conf, 'r', encoding='utf8') as f:
                transform_conf = json.load(f)
            pretrained_cfg['transform_config'] = transform_conf
        transform_train, transform_eval = TransformFactory.get_transform(hp.transform, **transform_conf)
        pretrained_cfg['transform'] = hp.transform
    model_config['pretrained_cfg'] = pretrained_cfg

    # set transform to dataset
    train_dataset.set_transform(partial(set_transform, transform=transform_train))
    eval_dataset.set_transform(partial(set_transform, transform=transform_eval))

    # initialize sampler
    train_sampler = RandomSampler(train_dataset, replacement=False)
    eval_sampler = RandomSampler(eval_dataset, replacement=False)

    # train model with subset of the entire dataset
    # control by hp.subset and hp.subset_ratio
    # hp.subset: subset, subset_sampler; use a fix subset or use a random subset
    # hp.subset_ratio: how many percentage of the dataset to use
    if hp.subset is not None:
        if hp.subset == 'subset':
            # train subset
            train_subset_indices = torch.randperm(len(train_dataset)).tolist()
            num_train_sample = np.ceil(hp.subset_ratio * len(train_dataset)).astype(int)
            train_dataset = Subset(train_dataset, train_subset_indices[:num_train_sample])
            # eval subset
            eval_subset_indices = torch.randperm(len(eval_dataset)).tolist()
            num_eval_sample = np.ceil(hp.subset_ratio * len(eval_dataset)).astype(int)
            eval_dataset = Subset(eval_dataset, eval_subset_indices[:num_eval_sample])
            # reinitialize sampler
            train_sampler = RandomSampler(train_dataset, replacement=False)
            eval_sampler = RandomSampler(eval_dataset, replacement=False)

        elif hp.subset == 'subset_sampler':
            num_train_sample = int(np.ceil(hp.subset_ratio * len(train_dataset)))
            train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=num_train_sample)
            num_eval_sample = int(np.ceil(hp.subset_ratio * len(eval_dataset)))
            eval_sampler = RandomSampler(eval_dataset, replacement=False, num_samples=num_eval_sample)
        else:
            raise ValueError("subset mode not supported")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

    hp_kwargs = vars(hp)
    hp_kwargs.pop('model')
    hp_kwargs.pop('num_classes')
    hp_kwargs.pop('pretrained')

    if hp.resume_model_path is not None:
        # resume training
        resume_model_path = hp.resume_model_path
        with open(os.path.join(resume_model_path, "config.json"), 'r', encoding='utf8') as f:
            model_config = json.load(f)
        model = load_model(resume_model_path)
        epochs -= model_config['trained_epoch']
    else:
        # create model
        model = generate_model(model_name, pretrained=pretrained, early_exits_gates=hp.ee, num_classes=num_classes,
                               **hp_kwargs)

    # initialize training procedure
    if hp.train_mode == 'freeze' and pretrained is None:
        raise ValueError("Please provide pretrained model path if you using freeze backbone training mode")

    train_mode = load_train_mode(hp.train_mode)
    # set training mode
    optimizer_config = {
        'module': set_optimizer(hp.optim),
        'lr': lr,
        'weight_decay': weight_decay,
        'momentum': momentum
    }

    # set scheduler
    scheduler_config = None
    if hp.scheduler is not None:
        scheduler_config = {
            'module': set_scheduler(hp.scheduler),
            'mode': hp.schedule_mode,
            'factor': hp.factor,
            'patience': hp.patience,
            'verbose': hp.verbose,
            'threshold': hp.threshold,
            'cooldown': hp.cooldown,
            'min_lr': hp.min_lr,
            'eps': hp.eps
        }

    # set save path
    model_save_path = os.path.join(os.path.split(dataset_cache_dir)[0], 'models',
                                   f"{model_name}_{dataset_name}_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    with open(os.path.join(model_save_path, "train_settings.json"), 'w', encoding='utf8') as f:
        json.dump(vars(hp), f, indent=4)
    with open(os.path.join(model_save_path, "config.json"), 'w', encoding='utf8') as f:
        json.dump(model_config, f, indent=4)

    # initialize training process
    train_process = train_mode(model, optimizer_config, scheduler_config, device=device, model_config=model_config,
                               model_save_path=model_save_path)
    try:
        train_process.fit(train_loader, eval_loader, epochs)
    except Exception as e:
        traceback.print_exc()
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument("--dataset", type=str, default="cifar",
                        help="Dataset to use. Options: ['cifar', 'imagenet']")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes in the dataset. cifar10: 10, cifar100: 100, imagenet: 1000")
    parser.add_argument("--test_split", type=str, default="test",
                        help="Dataset split for evaluation. Options: ['test', 'val']. cifar10: test, cifar100: test, imagenet: val")
    parser.add_argument("--subset", type=str, default=None,
                        help="Options: subset, subset_sampler")
    parser.add_argument("--subset_ratio", type=float, default=0.1,
                        help="Percentage of the dataset to use for subset")

    # arguments for model type
    parser.add_argument("--model", type=str, default="custom_9increase",
                        help="Model to use. Options: ['custom_9increase', 'resnet18', 'resnet34', 'resnet50', 'resnet101']")
    parser.add_argument("--ee", metavar='N', type=int, nargs='+',
                        help="indicate the index of early exit layers.")
    parser.add_argument("--transform", type=str, default='v1',
                        help="Transform type to use. Options: ['v1', 'v2', 'v3', 'config']")
    parser.add_argument("--transform_conf", type=str, default=None,
                        help="Transform configure file path. Only used when transform is set to 'config'")
    parser.add_argument("--confidence_score", type=str, default="max",
                        help="Confidence score function to use. Options: ['max', 'entropy']")

    # arguments for pretrained model
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Use pretrained model or not. Options: ['path_to_pretrained_model']")
    parser.add_argument("--resume_model_path", type=str, default=None,
                        help="Resume training from a model checkpoint. Options: ['path_to_model_checkpoint']")

    # arguments for model
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden_features", default=False, action="store_true")

    # arguments for optimizer
    parser.add_argument("--optim", type=str, default="SGD",
                        help="Optimizer to use. Options: ['SGD', 'Adam']")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)

    # arguments for learning rate scheduler
    parser.add_argument("--scheduler", type=str, default=None,
                        help="Learning rate scheduler to use. Options: ['ReduceLROnPlateau'] check pytorch documentation for more options")
    parser.add_argument("--schedule_mode", type=str, default="max")
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--threshold_mode", type=str, default="rel")
    parser.add_argument("--cooldown", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1e-8)

    # arguments for training
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_mode", type=str, default='naive',
                        help="Training mode for Early Exit Network. "
                             "Possible train mode: ['naive', 'end_weight', 'end_join, 'freeze', 'incremental']")
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--verbose", default=True, action="store_true")
    hp = parser.parse_args()

    main(hp)
