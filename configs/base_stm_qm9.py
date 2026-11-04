"""Auxiliary QM9-STM config for ``--config_original_qm9``: the reference molecules for the metrics."""

from configs.diffspectra_stm_qm9 import get_config as _main_config


def get_config():
    config = _main_config()
    config.data.load_images = False
    return config
