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
import shutil
import tempfile

import torch
from tqdm import tqdm
from urllib.request import Request, urlopen

import appdirs

APP_NAME = 'EENN'


def create_model_dir(model_name):
    cache_folder = appdirs.user_cache_dir(APP_NAME)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    model_folder = os.path.join(cache_folder, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    return model_folder


def download_model_config(url, model_name):
    model_folder = create_model_dir(model_name)
    file_name = os.path.join(model_folder, 'config.json')

    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    f = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(file_name))
    try:
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)
        f.close()
        shutil.move(f.name, file_name)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
    return file_name


def download_cached_file(url, model_name, progress=True):
    model_folder = create_model_dir(model_name)

    file_name = os.path.join(model_folder, 'model.pkl')
    if not os.path.exists(file_name):
        print(f"Download {model_name} from {url} and save to {file_name}")

        file_size = None
        req = Request(url, headers={"User-Agent": "torch.hub"})
        u = urlopen(req)
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            content_length = meta.getheaders("Content-Length")
        else:
            content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])

        f = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(file_name))

        try:
            with tqdm(total=file_size, disable=not progress,
                      unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))

            f.close()
            shutil.move(f.name, file_name)
        except Exception as e:
            print("[ERROR] Download model failure")
            print(e)
            file_name = None
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)
    return file_name


def save_model(model, model_name, post_fix, save_path=None):
    if save_path is not None:
        save_to_path = input(f"Do you want to save the model directory to {save_path}? (Y/N)")
        assert save_to_path.lower() in ['y', 'n'], 'Invalid selection break'
        if save_to_path.lower() == 'y':
            print(f"Save model to {save_path}")
        else:
            save_path = create_model_dir(f"{model_name}_{post_fix}")
            print(f"Save model to {save_path}")
    else:
        save_path = create_model_dir(f"{model_name}_{post_fix}")
        print(f"Save model to {save_path}")

    torch.save(model.state_dict(), f"{save_path}/model.pth")
