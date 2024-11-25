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

from tqdm import tqdm
import torch
import torch.nn as nn

from ..nn.model import EarlyExitsModel
from ..nn.modules.loss import JoinEarlyExitClassificationLoss, EarlyExitClassificationLoss, \
    SelectGateEarlyExitClassificationLoss

SUPPORT_MODE = ["naive", "end_weight", "end_join", "freeze", "freeze_origin", "incremental"]


class EpochEvalMetric:
    def __init__(self):
        self.sample_count = 0
        self.loss_cumulate = 0
        self.acc_cumulate = None
        self.best_possible = None

    @property
    def loss(self):
        return self.loss_cumulate / self.sample_count

    @property
    def accuracy(self):
        if self.acc_cumulate is None:
            return None
        return self.acc_cumulate / self.sample_count

    @property
    def best_possible_acc(self):
        if self.best_possible is None:
            return None
        return self.best_possible / self.sample_count

    def pbar_info(self):
        postfix_dict = {"loss": "%.5f" % self.loss}
        if self.acc_cumulate is not None:
            n_gate = self.acc_cumulate.shape[0]
            for i, acc in enumerate(self.accuracy):
                if i == n_gate - 1:
                    key = "Full model acc"
                else:
                    key = f"Gate {i + 1} acc"
                postfix_dict[key] = "%.5f" % acc
        return postfix_dict

    def eval_info(self, prefix=''):
        info_list = list()
        info_dict = self.pbar_info()
        for k, v in info_dict.items():
            info_list.append(f"{' '.join([k, prefix])}: {v}")
        info_str = " - ".join(info_list)
        return info_str

    def update(self, n_samples, loss, output, target):
        self.sample_count += n_samples
        self.loss_cumulate += loss.detach().cpu().numpy() * n_samples

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu()
            target = target.cpu()
            if output.dim() == 2:
                output = output.unsqueeze(dim=-1)

            if target.dim() == 1:
                target = target.unsqueeze(dim=-1)

            output_pred = torch.argmax(output, dim=1)
        else:
            output_pred = torch.tensor([[output]])
        acc = torch.eq(target, output_pred).sum(dim=0)
        best_possible = torch.eq(target, output_pred).max(dim=1)[0].sum(dim=0)
        if self.acc_cumulate is None:
            self.acc_cumulate = acc
            self.best_possible = best_possible
        else:
            self.acc_cumulate += acc
            self.best_possible += best_possible


class TrainProcess:
    def __init__(self,
                 model,
                 optimizer_config,
                 scheduler_config=None,
                 device='cpu',
                 model_config=None,
                 model_save_path=None,
                 **kwargs):
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.device = device
        self.current_epoch = 0
        self.model_config = model_config
        self.model_save_path = model_save_path

    def fit(self, train_dataloader, test_dataloader, epochs, **kwargs):
        raise NotImplementedError

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "model.pth"))
        self.model_config['trained_epoch'] = self.current_epoch
        with open(os.path.join(self.model_save_path, "config.json"), 'w', encoding='utf8') as f:
            json.dump(self.model_config, f, indent=4)

    def _train(self, train_dataloader, valid_dataloader, loss_func, optimizer, epochs, scheduler=None):
        self.model.to(self.device)
        loss_func = loss_func.to(self.device)
        for e in range(epochs):
            print(f"Epoch {e + 1}/{epochs}")
            self.model.train()
            train_metric = EpochEvalMetric()
            pbar = tqdm(train_dataloader)
            for batch in pbar:
                optimizer.zero_grad()
                if 'img' in batch.keys():
                    x, y = batch['img'], batch['label']
                else:
                    x, y = batch['image'], batch['label']
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = loss_func(y_hat, y)
                loss.backward()
                optimizer.step()

                train_metric.update(n_samples=x.shape[0], loss=loss, output=y_hat, target=y)

                pbar.set_postfix(train_metric.pbar_info())

            self.current_epoch += 1
            pbar.close()

            # Save model checkpoint after each epoch
            self._save_checkpoint()

            eval_metric = self._eval(valid_dataloader, loss_func)
            if scheduler:
                eval_loss = eval_metric.loss
                scheduler.step(eval_loss)

            print(
                "========== Epoch Summary ==========\n",
                f"Epoch: {e + 1}:\n",
                f"Train: {train_metric.eval_info()}\n",
                f"Eval: {eval_metric.eval_info()}\n"
            )
            with open(os.path.join(self.model_save_path, "train_log.txt"), 'a', encoding='utf8') as f:
                f.write(
                    "========== Epoch Summary ==========\n"
                    f"Epoch: {e + 1}:\n"
                    f"Train: {train_metric.eval_info()}\n"
                    f"Eval: {eval_metric.eval_info()}\n"
                )
            if hasattr(loss_func, "step"):
                loss_func.step()

    def _eval(self, valid_dataloader, loss_func):
        self.model.to(self.device)
        loss_func = loss_func.to(self.device)
        self.model.eval()
        eval_metric = EpochEvalMetric()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                if 'img' in batch.keys():
                    x, y = batch['img'], batch['label']
                else:
                    x, y = batch['image'], batch['label']
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = loss_func(y_hat, y)
                eval_metric.update(n_samples=x.shape[0], loss=loss, output=y_hat, target=y)
        return eval_metric

    def prepare_optimizer_scheduler(self):
        optimizer = create_optimizer(self.model.parameters(), self.optimizer_config)
        scheduler = create_scheduler(optimizer, self.scheduler_config)
        return optimizer, scheduler

    @staticmethod
    def disable_track_running_stats(modules):
        for model in modules.modules():
            if 'BatchNorm' in model._get_name():
                model.track_running_stats = False


class NaiveTrain(TrainProcess):
    def __init__(self,
                 model,
                 optimizer_config,
                 scheduler_config=None,
                 device='cpu',
                 model_config=None,
                 model_save_path=None,
                 **kwargs):
        assert not isinstance(model, EarlyExitsModel), ("Invalid model type. "
                                                        "This train process is not for early exit network")
        super(NaiveTrain, self).__init__(model, optimizer_config, scheduler_config, device, model_config,
                                         model_save_path, **kwargs)

    def fit(self, train_dataloader, test_dataloader, epochs, **kwargs):
        optimizer, scheduler = self.prepare_optimizer_scheduler()
        loss_func = nn.CrossEntropyLoss()
        self._train(train_dataloader, test_dataloader, loss_func, optimizer, epochs, scheduler)


class End2EndEarlyExitTrain(TrainProcess):
    def __init__(self,
                 model,
                 optimizer_config,
                 scheduler_config=None,
                 device='cpu',
                 model_config=None,
                 model_save_path=None,
                 **kwargs):
        assert isinstance(model, EarlyExitsModel), ("Invalid model type. "
                                                    "Your model should be generated from class 'EarlyExitsModel'")
        super(End2EndEarlyExitTrain, self).__init__(model, optimizer_config, scheduler_config, device, model_config,
                                                    model_save_path, **kwargs)

    def fit(self, train_dataloader, test_dataloader, epochs, **kwargs):
        optimizer, scheduler = self.prepare_optimizer_scheduler()
        loss_func = EarlyExitClassificationLoss(nn.CrossEntropyLoss(reduction='none'), len(self.model.gates_layer))
        self._train(train_dataloader, test_dataloader, loss_func, optimizer, epochs, scheduler)


class End2EndJoinEarlyExitTrain(TrainProcess):
    def __init__(self,
                 model,
                 optimizer_config,
                 scheduler_config=None,
                 device='cpu',
                 model_config=None,
                 model_save_path=None,
                 **kwargs
                 ):
        assert isinstance(model, EarlyExitsModel), ("Invalid model type. "
                                                    "Your model should be generated from class 'EarlyExitsModel'")
        super(End2EndJoinEarlyExitTrain, self).__init__(model, optimizer_config, scheduler_config, device, model_config,
                                                        model_save_path, **kwargs)

    def fit(self, train_dataloader, test_dataloader, epochs, **kwargs):
        optimizer, scheduler = self.prepare_optimizer_scheduler()
        loss_func = JoinEarlyExitClassificationLoss(nn.CrossEntropyLoss(reduction='none'), len(self.model.gates_layer))
        self._train(train_dataloader, test_dataloader, loss_func, optimizer, epochs, scheduler)


class FreezeBackboneEarlyExitTrain(TrainProcess):
    def __init__(self,
                 model,
                 optimizer_config,
                 scheduler_config=None,
                 device='cpu',
                 model_config=None,
                 model_save_path=None,
                 **kwargs
                 ):
        assert isinstance(model, EarlyExitsModel), ("Invalid model type. "
                                                    "Your model should be generated from class 'EarlyExitsModel'")
        super(FreezeBackboneEarlyExitTrain, self).__init__(model, optimizer_config, scheduler_config, device,
                                                           model_config, model_save_path ** kwargs)

    def fit(self, train_dataloader, test_dataloader, epochs, **kwargs):
        self.model.backbone.requires_grad_(False)
        self.disable_track_running_stats(self.model.backbone)
        optimizer, scheduler = self.prepare_optimizer_scheduler()
        loss_func = EarlyExitClassificationLoss(nn.CrossEntropyLoss(reduction='none'), len(self.model.gates_layer))
        self._train(train_dataloader, test_dataloader, loss_func, optimizer, epochs, scheduler)


class FreezeOriginModelEarlyExitTrain(TrainProcess):
    def __init__(self,
                 model,
                 optimizer_config,
                 scheduler_config=None,
                 device='cpu',
                 model_config=None,
                 model_save_path=None,
                 **kwargs
                 ):
        assert isinstance(model, EarlyExitsModel), ("Invalid model type. "
                                                    "Your model should be generated from class 'EarlyExitsModel'")
        super(FreezeOriginModelEarlyExitTrain, self).__init__(model, optimizer_config, scheduler_config, device,
                                                              model_config, model_save_path, **kwargs)

    def fit(self, train_dataloader, test_dataloader, epochs, **kwargs):
        self.model.backbone.requires_grad_(False)
        self.model.gates[str(self.model.gates_layer[-1])].requires_grad_(False)
        self.disable_track_running_stats(self.model.backbone)
        self.disable_track_running_stats(self.model.gates[str(self.model.gates_layer[-1])])
        optimizer, scheduler = self.prepare_optimizer_scheduler()
        loss_func = EarlyExitClassificationLoss(nn.CrossEntropyLoss(reduction='none'), len(self.model.gates_layer))
        self._train(train_dataloader, test_dataloader, loss_func, optimizer, epochs, scheduler)


class IncrementalEarlyExitTrain(TrainProcess):
    def __init__(self,
                 model,
                 optimizer_config,
                 scheduler_config=None,
                 device='cpu',
                 model_config=None,
                 model_save_path=None,
                 **kwargs
                 ):
        assert isinstance(model, EarlyExitsModel), ("Invalid model type. "
                                                    "Your model should be generated from class 'EarlyExitsModel'")
        super(IncrementalEarlyExitTrain, self).__init__(model, optimizer_config, scheduler_config, device, model_config,
                                                        model_save_path, **kwargs)

    def fit(self, train_dataloader, test_dataloader, epochs, **kwargs):
        prev_get_pos = 0
        for i, gate_pos in enumerate(self.model.gates_layer):
            optim_dict = [
                {"params": self.model.backbone[prev_get_pos:gate_pos + 1].parameters()},
                {"params": self.model.gates[str(gate_pos)].parameters()}
            ]
            optimizer = create_optimizer(optim_dict, self.optimizer_config.copy())
            scheduler = create_scheduler(optimizer, self.scheduler_config)
            print(f"==========Train Block {i + 1}/{len(self.model.gates_layer)}==========")
            loss_func = SelectGateEarlyExitClassificationLoss(nn.CrossEntropyLoss(), gate_id=i)
            self._train(train_dataloader, test_dataloader, loss_func, optimizer, epochs, scheduler)
            self.model.backbone[prev_get_pos:gate_pos + 1].requires_grad_(False)
            self.model.gates[str(gate_pos)].requires_grad_(False)
            prev_get_pos = gate_pos + 1


def create_optimizer(parameters, optimizer_config):
    optimizer_module = optimizer_config.pop('module')
    valid_args = optimizer_module.__init__.__code__.co_varnames[1:optimizer_module.__init__.__code__.co_argcount]
    filtered_dict = {k: v for k, v in optimizer_config.items() if k in valid_args}
    optimizer = optimizer_module(
        parameters,
        **filtered_dict
    )
    return optimizer


def create_scheduler(optimizer, scheduler_config):
    scheduler = None
    if scheduler_config:
        scheduler_module = scheduler_config.pop('module')
        valid_args = scheduler_module.__init__.__code__.co_varnames[1:scheduler_module.__init__.__code__.co_argcount]
        filtered_dict = {k: v for k, v in scheduler_config.items() if k in valid_args}
        scheduler = scheduler_module(
            optimizer,
            **filtered_dict
        )
    return scheduler


def load_train_mode(mode: str):
    mode = mode.lower()
    assert mode in SUPPORT_MODE, f"Unsupported training mode. Please select from the following {SUPPORT_MODE}"
    if mode == 'naive':
        mode_module = NaiveTrain
    elif mode == 'end_weight':
        mode_module = End2EndEarlyExitTrain
    elif mode == 'end_join':
        mode_module = End2EndJoinEarlyExitTrain
    elif mode == 'freeze':
        mode_module = FreezeBackboneEarlyExitTrain
    elif mode == 'freeze_origin':
        mode_module = FreezeOriginModelEarlyExitTrain
    elif mode == 'incremental':
        mode_module = IncrementalEarlyExitTrain
    else:
        raise ValueError
    return mode_module
