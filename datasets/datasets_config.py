"""Per-dataset element sets and the reference statistics used by the evaluation metrics.

``atom_encoder`` order must match the graph files' ``atom_type`` indices. ``name`` selects the
valence rules in ``evaluation/stability.py``.
"""

QM9 = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'atom_fc_num': {'N1': 20738, 'N-1': 8024, 'C1': 4117, 'O-1': 192, 'C-1': 764},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H',
                      'C2C-C1C'],
    'top_dihedral_sym': ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O',
                         'C1N-N1C-C1C', 'H1C-C1N-N1C', 'H1C-C1C-C1N'],
}

DATASET_INFO = {'qm9_second_half': QM9}


def get_dataset_info(info_name):
    if info_name not in DATASET_INFO:
        raise KeyError(f'unknown data.info_name {info_name!r}; known: {sorted(DATASET_INFO)}')
    return DATASET_INFO[info_name]
