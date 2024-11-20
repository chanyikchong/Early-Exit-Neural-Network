import numpy as np
from torch.utils.data import Subset

from .utils import *

__all__ = ["load_dataset"]

SUPPORT_DATASETS = [
    "imagenet_1k",
    "cifar_10"
]


def load_dataset(dataset_name, huggingface_token=None, split='test', streaming=True, cache_dir=None,
                 valid_mode=None, seed=0, valid_ratio=0.5):
    f"""
    Load DATASET from dataset name
    Current support dataset names:
    {SUPPORT_DATASETS}
    :param dataset_name: The name of the dataset
    :param huggingface_token: huggingface token to login into hugging face
    :param split: load partition of the dataset ['train', 'test', 'validate']
    :param streaming: load dataset in streaming process. Avoid download the complete dataset
    :param cache_dir
    
    The following parameters are used for further splitting the validation dataset
    :param valid_mode: further splitting test or validation set
    :param seed: random seed for splitting
    :param valid_ratio: ratio of validation set
    :return: Dataset
    """
    assert dataset_name.lower() in SUPPORT_DATASETS, f"{dataset_name} not supported"

    dataset = load_image_dataset(dataset_name.lower(), huggingface_token, split, streaming, cache_dir=cache_dir)
    if valid_mode is not None:
        len_dataset = len(dataset)
        np.random.seed(seed)
        indices = np.random.permutation(len_dataset).tolist()
        valid_size = int(len_dataset * valid_ratio)
        if valid_mode == 'valid':
            dataset = Subset(dataset, indices[:valid_size])
        elif valid_mode == 'test':
            dataset = Subset(dataset, indices[valid_size:])
        else:
            raise ValueError(f"valid_mode {valid_mode} not supported")
        setattr(dataset, 'set_transform', dataset.dataset.set_transform)
    return dataset
