"""Transforms for images. An image is a (C, H, W) tensor: C channels of H x W pixels."""

import math

import torch
from torchvision.transforms import v2


class PerChannelNormalize:
    """TERS: standardise each channel (one frequency) on its own."""

    def __call__(self, image):
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True)
        std[std < 1e-6] = 1.0
        return (image - mean) / std


class StmNormalize:
    """STM: scale uint16 to [0, 1], compress bright pixels with asinh, standardise the whole stack."""

    def __call__(self, image):
        x = torch.asinh(image.float() / 65535.0 / 0.01)
        std = x.std()
        if std < 1e-6:
            std = 1.0
        return (x - x.mean()) / std


class RandomSlice:
    """Keep one random z-slice: (Z, H, W) -> (1, H, W)."""

    def __call__(self, image):
        return image[torch.randint(len(image), ()).item()][None]


class FixedSlice:
    """Keep z-slice `index`: (Z, H, W) -> (1, H, W)."""

    def __init__(self, index=0):
        self.index = index

    def __call__(self, image):
        return image[self.index][None]


def get_transforms(single_slice=False, hflip_rot90=False, rotate_deg=0, translate=0,
                   blur_sigma=0, noise_std=0, random_erasing=0, p=0.5, eval_noise_std=0):
    """Return (train_transforms, eval_transforms).

    Every augmentation is off by default and is used for training only. Each enabled one is
    applied with probability `p`; `random_erasing` is its own probability. `translate` is a
    fraction of the image size. `eval_noise_std` adds noise to every train and eval image, to
    test how robust the model is to noisy images.
    """
    train_transforms, eval_transforms = [], []
    if single_slice:
        train_transforms.append(RandomSlice())
        eval_transforms.append(FixedSlice(0))

    if hflip_rot90:
        quarter_turns = [v2.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]
        train_transforms += [v2.RandomHorizontalFlip(), v2.RandomChoice(quarter_turns)]
    if rotate_deg:
        rotation = v2.RandomRotation(rotate_deg, interpolation=v2.InterpolationMode.BILINEAR)
        train_transforms.append(v2.RandomApply([rotation], p=p))
    if translate:
        translation = v2.RandomAffine(degrees=0, translate=(translate, translate))
        train_transforms.append(v2.RandomApply([translation], p=p))
    if blur_sigma:
        blur = v2.GaussianBlur(2 * math.ceil(3 * blur_sigma) + 1, sigma=(0.1, blur_sigma))
        train_transforms.append(v2.RandomApply([blur], p=p))
    if noise_std:
        noise = v2.GaussianNoise(sigma=noise_std, clip=False)
        train_transforms.append(v2.RandomApply([noise], p=p))
    if random_erasing:
        train_transforms.append(v2.RandomErasing(p=random_erasing, scale=(0.02, 0.1), ratio=(1, 1)))
    if eval_noise_std:
        train_transforms.append(v2.GaussianNoise(sigma=eval_noise_std, clip=False))
        eval_transforms.append(v2.GaussianNoise(sigma=eval_noise_std, clip=False))

    return (v2.Compose(train_transforms or [v2.Identity()]),
            v2.Compose(eval_transforms or [v2.Identity()]))
