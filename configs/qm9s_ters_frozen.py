"""QM9S-TERS encoder trained against the frozen, pretrained DiffSpectra denoiser; run with main_finetune.py."""

from configs.diffspectra_ters_qm9s import get_config as _main_config
from env import data_root


def get_config():
    config = _main_config()
    config.data.bonds = 'qm9'                  # the denoiser was trained on QM9's SDF graphs
    config.data.num_workers = 16
    config.model.pretrained_denoiser_path = data_root(
        'code', 'DiffSpectra', 'exp', 'allspectra_pretrained_specformer', 'checkpoints', 'checkpoint_40.pth')
    config.model.load_pretrained_cond_encoder = False
    config.model.dropout = 0.0                 # dropout only acts inside the frozen denoiser
    config.training.n_iters = 60000 // config.training.num_gpus
    config.training.snapshot_freq = 5000 // config.training.num_gpus
    config.eval.begin_ckpt = 12
    config.eval.end_ckpt = 12
    config.eval.num_samples = 782
    config.training.wandb.group = 'qm9s-frozen'
    return config
