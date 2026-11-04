"""STM dataset: ``TersDataset`` reading z-stacks from ``'<split>/X'`` and normalising with
``StmNormalize``. The image is stored under ``data.stm`` for ``build_dataset.CollateStm``."""

import h5py

from .transforms import StmNormalize
from .ters_dataset import TersDataset

IMAGE_KEY = 'X'


def get_image_key(h5_path, split=None, image_key=IMAGE_KEY):
    """Return ``'<split>/X'``; the split is inferred when the file holds a single group."""
    with h5py.File(h5_path, 'r') as f:
        if split is None:
            if image_key in f:
                return image_key
            groups = [k for k in f if isinstance(f[k], h5py.Group)]
            if len(groups) != 1:
                raise KeyError(f'{h5_path}: pass split=, found groups {groups}')
            split = groups[0]
        key = f'{split}/{image_key}'
        if key not in f:
            raise KeyError(f'{h5_path} has no dataset {key!r}; keys: {list(f)}')
        return key


class StmDataset(TersDataset):
    """One split of an STM corpus; see ``TersDataset`` for the arguments.

    ``normalize=True`` uses ``StmNormalize``. ``PerChannelNormalize`` would erase the signal
    decay from one z-slice to the next.
    """

    image_attr = 'stm'

    def __init__(self, h5_path, graphs_path, split=None, transform=None, normalize=True,
                 load_images=True, image_key=IMAGE_KEY, noise=None):
        if load_images:
            image_key = get_image_key(h5_path, split, image_key)
        if normalize is True:
            normalize = StmNormalize()
        super().__init__(h5_path, graphs_path, image_key, transform, normalize, load_images, noise)
        self.z_slices = self.get_z_slices(h5_path) if load_images else None

    def get_z_slices(self, h5_path):
        """Tip heights from the file, checked against the number of image channels."""
        with h5py.File(h5_path, 'r') as f:
            z_slices = f[self.image_key].parent.attrs.get('z_slices')
        if z_slices is not None and len(z_slices) != self.in_channels:
            raise ValueError(f'{h5_path}: z_slices lists {len(z_slices)} heights but the '
                             f'images have {self.in_channels} channels')
        return None if z_slices is None else list(z_slices)
