from torchvision import transforms as T
import timm.data


def set_transform(examples, transform):
    try:
        examples['img'] = [transform(i) for i in examples['img']]
    except KeyError:
        examples['image'] = [transform(i) for i in examples['image']]
    return examples


def convert_color(img, color_type):
    if img.mode != color_type:
        return img.convert(color_type)
    return img


'''
timm.data.create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        crop_mode=None,
        tf_preprocessing=False,
        separate=False)
'''


def create_transform(model, is_training=False, color_type=None):
    assert model.pretrained_cfg, "The model does not contain attribute 'pretrain_config'"
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=is_training)
    if color_type:
        transforms.transforms = [T.Lambda(lambda img: convert_color(img, color_type))] + transforms.transforms
    return transforms