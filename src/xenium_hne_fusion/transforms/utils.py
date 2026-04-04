from timm.data import create_transform, resolve_data_config
from torchvision.transforms import v2


def get_timm_transform(model):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config, is_training=False)
    _validate_center_crop(transform)
    return transform


def get_normalize_from_transform(transform):
    (normalize,) = [t for t in transform.transforms if t.__class__.__name__ == 'Normalize']
    return v2.Normalize(mean=normalize.mean, std=normalize.std)


def _validate_center_crop(transform):
    for t in transform.transforms:
        if t.__class__.__name__ == 'CenterCrop':
            assert t.size == (224, 224), f'Expected CenterCrop (224, 224), got {t.size}'
