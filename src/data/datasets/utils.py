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

from datasets import load_dataset
import huggingface_hub

__all__ = ['load_image_dataset']

HUGGINGFACE_CACHE_TOKEN = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'token')


def huggingface_login(huggingface_token=None):
    if not os.path.exists(HUGGINGFACE_CACHE_TOKEN):
        if huggingface_token is None:
            raise ValueError("Please provide a huggingface token and save it in `~/.cache/huggingface/token`. You can get a token by running `huggingface-cli login`")
        huggingface_hub.login(huggingface_token)
    else:
        with open(HUGGINGFACE_CACHE_TOKEN) as f:
            token = f.read().strip()
            if not token:
                raise ValueError("Please provide a huggingface token and save it in `~/.cache/huggingface/token`. You can get a token by running `huggingface-cli login`")
        huggingface_hub.login(token)


def load_image_dataset(dataset_name, huggingface_token=None, split='test', streaming=True, cache_dir=None):
    huggingface_login(huggingface_token)
    script_path = os.path.dirname(__file__)
    dataset_path = os.path.join(script_path, dataset_name)
    dataset = load_dataset(dataset_path, split=split, streaming=streaming, cache_dir=cache_dir)
    return dataset
