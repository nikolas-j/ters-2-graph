"""Datasets and data loaders for image-conditioned molecule generation."""

import os.path as osp

import h5py
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2

from .datasets_config import get_dataset_info
from .image_dataset import ImageGraphDataset
from .transforms import PerChannelNormalize, get_transforms


def get_dataset(config):
    """Return ``(train, train, val, test, dataset_info)``; the train set fills both train slots."""
    data = config.data
    info = get_dataset_info(data.info_name)
    transform = EdgeComSpectraTransform(info['atom_encoder'].values(), data.include_aromatic)
    table = Chem.GetPeriodicTable()
    atom_types = {table.GetAtomicNumber(symbol): index for symbol, index in info['atom_encoder'].items()}

    train = _build_split(data, 'train', transform, atom_types)
    val = _build_split(data, 'val', transform, atom_types)
    test = _build_split(data, 'test', transform, atom_types)
    if getattr(data, 'transforms', None):
        train.transform_image, val.transform_image = get_transforms(**data.transforms)
        test.transform_image = val.transform_image
    return train, train, val, test, info


def _build_split(data, split, transform, atom_types):
    h5_path = osp.join(data.root, f'{split}.h5')
    smiles_path = None
    if getattr(data, 'smiles_root', ''):
        smiles_path = osp.join(data.smiles_root, f'{split}.h5')
    return ImageGraphDataset(h5_path, data.cache_dir, data.image_key, atom_types, bonds=data.bonds,
                             qm9_path=getattr(data, 'qm9_path', None), smiles_path=smiles_path,
                             transform=transform, normalize=_get_normalizer(data, h5_path),
                             load_images=getattr(data, 'load_images', True))


def _get_normalizer(data, h5_path):
    """``data.normalize``: 'per_channel' (TERS), 'corpus' (one mean/std from the file's attributes, STM)."""
    if data.normalize == 'per_channel':
        return PerChannelNormalize()
    if data.normalize == 'corpus':
        with h5py.File(h5_path, 'r') as f:
            return v2.Normalize(mean=[float(f.attrs['mean'])], std=[float(f.attrs['std'])])
    raise ValueError(f"data.normalize must be 'per_channel' or 'corpus', got {data.normalize!r}")


def get_dataloader(train_ds, val_ds, test_ds, config):
    """Return ``(train, val, test)`` loaders. Under DDP, ``training.batch_size`` is per rank."""
    if config.only_2D:
        raise ValueError('only_2D is not supported: the image conditions the 3D positions')
    workers = getattr(config.data, 'num_workers', 2)
    options = dict(num_workers=workers, collate_fn=Collate(train_ds.image_attr))
    if workers > 0:
        options['prefetch_factor'] = getattr(config.data, 'prefetch_factor', 2)
    sampler = None
    if getattr(config.training, 'ddp', False):
        import distributed
        sampler = DistributedSampler(train_ds, distributed.get_world_size(), distributed.get_rank(),
                                     shuffle=True, drop_last=config.training.dataloader_drop_last)
    train = DataLoader(train_ds, config.training.batch_size, shuffle=sampler is None, sampler=sampler,
                       drop_last=config.training.dataloader_drop_last, persistent_workers=workers > 0,
                       **options)
    val = DataLoader(val_ds, config.training.eval_batch_size, **options)
    test = DataLoader(test_ds, config.training.eval_batch_size, **options)
    return train, val, test


def inf_iterator(loader):
    """Yield batches forever, reshuffling each epoch (``set_epoch`` for the DDP sampler)."""
    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


class EdgeComSpectraTransform:
    """Add ``atom_one_hot`` (N, types) and dense ``edge_one_hot`` (N, N, C) to a molecule.

    Edge channels: bond exists; bond order / 3 (aromatic counts as 0); aromatic, if
    ``include_aromatic``. Kekulized graphs have no aromatic bonds.
    """

    def __init__(self, atom_types, include_aromatic):
        self.atom_types = torch.tensor(list(atom_types))
        self.include_aromatic = include_aromatic

    def __call__(self, data):
        data.atom_one_hot = (data.atom_type[:, None] == self.atom_types[None]).float()
        aromatic = data.edge_type == 4
        features = [torch.where(aromatic, 0, data.edge_type) / 3]
        if self.include_aromatic:
            features.append(aromatic.float())
        features = torch.stack(features, dim=-1)
        dense = torch.zeros(data.num_nodes, data.num_nodes, features.size(-1))
        dense.index_put_((data.edge_index[0], data.edge_index[1]), features, accumulate=True)
        exists = (dense.sum(-1, keepdim=True) != 0).float()
        data.edge_one_hot = torch.cat([exists, dense], dim=-1)
        return data


class Collate:
    """Pad molecules to the largest in the batch and stack their images into ``context``."""

    def __init__(self, image_attr):
        self.image_attr = image_attr

    def __call__(self, items):
        n = max(int(item.num_atom) for item in items)
        atom_mask = torch.stack([torch.arange(n) < int(item.num_atom) for item in items]).float()
        edge_mask = atom_mask[:, :, None] * atom_mask[:, None, :] * (1 - torch.eye(n))
        return dict(
            atom_one_hot=_pad_nodes([item.atom_one_hot for item in items], n),
            edge_one_hot=_pad_edges([item.edge_one_hot for item in items], n),
            positions=_pad_nodes([item.pos for item in items], n),
            formal_charges=_pad_nodes([item.fc[:, None] for item in items], n),
            atom_mask=atom_mask,
            edge_mask=edge_mask.reshape(-1, 1),
            context=torch.stack([getattr(item, self.image_attr) for item in items]),
        )


def _pad_nodes(tensors, n):
    return torch.stack([F.pad(x, (0, 0, 0, n - len(x))) for x in tensors])


def _pad_edges(tensors, n):
    return torch.stack([F.pad(x, (0, 0, 0, n - len(x), 0, n - len(x))) for x in tensors])
