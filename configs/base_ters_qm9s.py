"""Auxiliary QM9S-TERS config for ``--config_original_qm9``: the reference molecules for the metrics.

The main config's dataset without images, so the two always describe the same molecules.
"""

from configs.diffspectra_ters_qm9s import get_config as _main_config


def get_config():
    config = _main_config()
    config.data.load_images = False
    return config
