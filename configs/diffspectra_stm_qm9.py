"""DiffSpectra conditioned on QM9-STM images: 11 constant-current setpoints of 64 x 64 pixels.

The QM9S-TERS config with the QM9-STM data; pair with ``--config_original_qm9 configs/base_stm_qm9.py``.
Training uses one random setpoint per sample and evaluation setpoint 0 (tip closest), so the encoder
has one input channel.
"""

from configs.diffspectra_ters_qm9s import QM9S_ROOT
from configs.diffspectra_ters_qm9s import get_config as _ters_config
from env import data_root

QM9_STM_ROOT = data_root('data', 'qm9-stm', 'h5')
QM9_STM_CACHE = data_root('data', 'qm9-stm')            # graphs are cached in <cache>/processed/


def get_config():
    config = _ters_config()
    data = config.data
    data.root = QM9_STM_ROOT                  # {train,val,test}.h5
    data.image_key = 'X'                      # (N, 64, 64, 11)
    data.normalize = 'corpus'                 # one mean/std for every setpoint keeps their differences
    data.cache_dir = QM9_STM_CACHE
    data.smiles_root = QM9S_ROOT              # QM9-STM rows reach their SMILES through ters_row
    data.transforms.single_slice = True
    config.model.ters_in_channels = 1
    config.training.wandb.group = 'qm9-stm'
    return config
