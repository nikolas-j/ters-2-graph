"""TERS dataset: graph records from a sidecar file, conditioning images from an HDF5 file.

Each sample is a PyG ``Data`` with the graph fields of ``preprocess_ters.py`` and the image as a
``(C, H, W)`` tensor under ``data.ters``. ``build_dataset.CollateTers`` batches the samples.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .transforms import PerChannelNormalize


class TersDataset(Dataset):
    """One split of a TERS corpus.

    Args:
        h5_path: HDF5 file holding the images as ``(N, H, W, C)`` under ``image_key``.
        graphs_path: sidecar ``.pt`` file written by ``preprocess_ters.py``.
        transform: graph transform, e.g. ``EdgeComSpectraTransform``.
        normalize: ``True`` for ``PerChannelNormalize``, ``False`` for none, or a callable.
        load_images: ``False`` returns graphs only, for the reference-molecule config.
        noise: ``FixedNoise`` applied after normalisation.
    """

    image_attr = 'ters'

    def __init__(self, h5_path, graphs_path, image_key='channels_100', transform=None,
                 normalize=True, load_images=True, noise=None):
        self.h5_path = h5_path
        self.image_key = image_key
        self.transform = transform
        self.normalize = normalize if callable(normalize) else PerChannelNormalize() if normalize else None
        self.noise = noise
        self.transform_image = None             # augmentation, set by build_dataset
        self.load_images = load_images
        self.image_shape = None
        self.in_channels = None
        self._hf = None                         # HDF5 handle, opened inside each worker

        self.graphs = torch.load(graphs_path, weights_only=False)   # holds RDKit Mols
        if not self.graphs:
            raise ValueError(f'{graphs_path} contains no records')
        if load_images:
            self.check_images(h5_path, graphs_path)

    def check_images(self, h5_path, graphs_path):
        with h5py.File(h5_path, 'r') as f:
            if self.image_key not in f:
                raise KeyError(f'{h5_path} has no dataset {self.image_key!r}; keys: {list(f)}')
            shape = f[self.image_key].shape
        if len(shape) != 4:
            raise ValueError(f'{self.image_key} has shape {shape}, expected (N, H, W, C)')
        self.image_shape = tuple(shape[1:])
        self.in_channels = shape[3]
        last_index = max(int(g['h5_index']) for g in self.graphs)
        if last_index >= shape[0]:
            raise IndexError(f'{graphs_path} refers to image {last_index}, but {h5_path} '
                             f'holds {shape[0]}: not the same split')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        data = Data(**graph)
        if self.load_images:
            if self._hf is None:
                self._hf = h5py.File(self.h5_path, 'r')
            index = int(graph['h5_index'])
            image = torch.from_numpy(np.asarray(self._hf[self.image_key][index])).float()
            image = image.permute(2, 0, 1).contiguous()                     # (H, W, C) -> (C, H, W)
            if self.normalize is not None:
                image = self.normalize(image)
            if self.noise is not None:
                image = self.noise(image, index)
            if self.transform_image is not None:
                image = self.transform_image(image)
            setattr(data, self.image_attr, image)
        return self.transform(data) if self.transform is not None else data

    def __getstate__(self):
        return {**self.__dict__, '_hf': None}    # never pickle an open HDF5 handle

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def __del__(self):
        try:
            self.close()
        except Exception:               # h5py may already be torn down at interpreter exit
            pass
