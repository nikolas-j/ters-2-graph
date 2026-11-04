"""Molecule graphs paired with HDF5 images (QM9S-TERS, QM9-STM, planar TERS).

Each HDF5 file holds the images and every molecule's atoms and coordinates. Graphs are built once, like DiffSpectra's ``QM9S.process``, and cached in
``<cache_dir>/processed/``; images are read from the HDF5 file on demand. ``bonds`` selects where
the bonds come from:

- ``'geometry'``: found from the HDF5 coordinates with RDKit, or taken from the molecule's SMILES
  when that fails. Molecules without a valid neutral, closed-shell graph are dropped.
- ``'qm9'``: DiffSpectra's ``data_qm9.pt`` (QM9's SDF), matched by the file's ``qm9_idx``. Needed to
  fine-tune the pretrained DiffSpectra denoiser, which was trained on these graphs.
"""

import multiprocessing
import os
import os.path as osp
from itertools import repeat

import h5py
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from torch_geometric.data import Data, InMemoryDataset

QM9_ATOM_TYPES = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}       # H, C, N, O, F, the atom_type indices in data_qm9.pt
BOND_TYPES = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
BOND_TOLERANCE = 0.1                                    # Å; a larger mismatch means a different molecule


class ImageGraphDataset(InMemoryDataset):
    """One split. ``image_key`` names the ``(N, H, W, C)`` images in ``h5_path``; ``atom_types`` maps
    atomic number to atom-type index.

    ``smiles_path`` is a QM9S-TERS HDF5 file of the same split; QM9-STM files have no SMILES and
    reach them through their ``ters_row``.
    """

    image_attr = 'image'

    def __init__(self, h5_path, cache_dir, image_key, atom_types, bonds='geometry', qm9_path=None,
                 smiles_path=None, transform=None, normalize=None, load_images=True):
        self.h5_path = h5_path
        self.image_key = image_key
        self.atom_types = atom_types
        self.bonds = bonds
        self.qm9_path = qm9_path
        self.smiles_path = smiles_path
        self.normalize = normalize
        self.load_images = load_images
        self.transform_image = None             # augmentation, set by build_dataset
        self._hf = None                         # HDF5 handle, opened once per process
        self._pid = None
        super().__init__(cache_dir, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        split = osp.basename(self.h5_path).replace('.h5', '')
        return [f'{split}_{self.bonds}.pt']

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        with h5py.File(self.h5_path, 'r') as f:
            qm9_idx = None
            if 'qm9_idx' in f:
                qm9_idx = f['qm9_idx'][:]
            molecules = _read_molecules(f)
            smiles = _read_smiles(f, self.smiles_path, len(molecules))

        if self.bonds == 'geometry':
            arguments = zip(molecules, smiles, repeat(self.atom_types))
            with multiprocessing.Pool(len(os.sched_getaffinity(0))) as pool:
                graphs = pool.starmap(_graph_from_geometry, arguments, chunksize=256)
        elif self.bonds == 'qm9':
            if qm9_idx is None or self.atom_types != QM9_ATOM_TYPES:
                raise ValueError("bonds='qm9' needs qm9_idx in the HDF5 file and the QM9 atom types")
            graphs = _graphs_from_qm9(self.qm9_path, qm9_idx, molecules)
        else:
            raise ValueError(f"bonds must be 'geometry' or 'qm9', got {self.bonds!r}")

        data_list = []
        for row, graph in enumerate(graphs):
            if graph is None:
                continue
            for key in ('atom_type', 'charge', 'fc', 'edge_index', 'edge_type', 'pos'):
                graph[key] = torch.from_numpy(graph[key])
            graph['qm9_idx'] = -1
            if qm9_idx is not None:
                graph['qm9_idx'] = int(qm9_idx[row])
            data_list.append(Data(**graph, h5_index=row))
        print(f'{self.h5_path}: kept {len(data_list)} of {len(graphs)} molecules')
        self.save(data_list, self.processed_paths[0])

    def get(self, idx):
        data = super().get(idx)
        if not self.load_images:
            return data
        if self._pid != os.getpid():
            self._hf = h5py.File(self.h5_path, 'r')
            self._pid = os.getpid()
        image = torch.from_numpy(self._hf[self.image_key][int(data.h5_index)]).float().permute(2, 0, 1)
        if self.normalize is not None:
            image = self.normalize(image)
        if self.transform_image is not None:
            image = self.transform_image(image)
        data.image = image
        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hf'] = None                     # an open HDF5 handle cannot be pickled
        state['_pid'] = None
        return state


def _read_molecules(f):
    """Atomic numbers and positions (Å) per row: QM9-STM stores ``xyz``, TERS files one group per row."""
    molecules = []
    if 'xyz' in f:
        for rows in f['xyz'][:]:
            atoms = rows[:, 3] > 0
            molecules.append((rows[atoms, 3].astype(int), rows[atoms, :3].astype(np.float64)))
    else:
        for i in range(len(f['atomic_numbers'])):
            numbers = f['atomic_numbers'][str(i)][()].astype(int)
            molecules.append((numbers, f['atom_positions'][str(i)][()].astype(np.float64)))
    return molecules


def _read_smiles(f, smiles_path, n):
    if 'smiles' in f:
        return list(f['smiles'].asstr()[:])
    if smiles_path is None:
        return [None] * n
    with h5py.File(smiles_path, 'r') as g:
        smiles = g['smiles'].asstr()[:]
    return [smiles[row] for row in f['ters_row'][:]]


def _graph_from_geometry(molecule, smiles, atom_types):
    numbers, pos = molecule
    if not set(numbers.tolist()) <= set(atom_types):
        return None
    mol = _bonds_from_geometry(numbers, pos)
    bonds_from = 'geometry'
    if mol is None and smiles is not None:
        mol = _bonds_from_smiles(numbers, pos, smiles)
        bonds_from = 'smiles'
    if mol is None:
        return None
    graph = _encode(mol, pos, atom_types)
    graph['bonds_from'] = bonds_from
    return graph


def _bonds_from_geometry(numbers, pos):
    mol = _mol_from_xyz(numbers, pos)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    if not _is_valid(mol):
        return None
    return mol


def _bonds_from_smiles(numbers, pos, smiles):
    """Bonds, bond orders and charges from the SMILES, mapped onto the coordinates' atoms.

    In strained cages some non-bonded atoms are closer than the bonding cutoff, so the distance
    graph has extra bonds. The SMILES skeleton is matched into that graph (shortest bonds when
    several matches exist) and only its bonds are kept; hydrogens keep their geometric bonds.
    """
    mol = _mol_from_xyz(numbers, pos)
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None
    rdDetermineBonds.DetermineConnectivity(mol)
    Chem.Kekulize(template, clearAromaticFlags=True)
    skeleton = Chem.RWMol(template)
    for atom in skeleton.GetAtoms():
        atom.SetFormalCharge(0)
    for bond in skeleton.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
    matches = mol.GetSubstructMatches(skeleton, uniquify=False, maxMatches=10000)
    if not matches:
        return None
    lengths = [_total_bond_length(template, match, pos) for match in matches]
    match = matches[int(np.argmin(lengths))]

    bond_types = {}
    for bond in template.GetBonds():
        bond_types[frozenset((match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]))] = bond.GetBondType()
    edited = Chem.RWMol(mol)
    for bond in list(edited.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if edited.GetAtomWithIdx(i).GetAtomicNum() == 1 or edited.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        if frozenset((i, j)) in bond_types:
            bond.SetBondType(bond_types[frozenset((i, j))])
        else:
            edited.RemoveBond(i, j)
    for atom in template.GetAtoms():
        edited.GetAtomWithIdx(match[atom.GetIdx()]).SetFormalCharge(atom.GetFormalCharge())
    mol = edited.GetMol()
    if not _is_valid(mol):
        return None
    return mol


def _total_bond_length(template, match, pos):
    total = 0.0
    for bond in template.GetBonds():
        total += np.linalg.norm(pos[match[bond.GetBeginAtomIdx()]] - pos[match[bond.GetEndAtomIdx()]])
    return total


def _mol_from_xyz(numbers, pos):
    table = Chem.GetPeriodicTable()
    lines = [f'{len(numbers)}', '']
    for z, (x, y, w) in zip(numbers, pos):
        lines.append(f'{table.GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}')
    return Chem.MolFromXYZBlock('\n'.join(lines))


def _is_valid(mol):
    """Every atom satisfied by its explicit bonds and charge: no added hydrogens, no radicals, net charge 0."""
    strict = Chem.Mol(mol)
    for atom in strict.GetAtoms():
        atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(strict)
    except Exception:
        return False
    if any(atom.GetNumRadicalElectrons() for atom in strict.GetAtoms()):
        return False
    return Chem.GetFormalCharge(strict) == 0


def _encode(mol, pos, atom_types):
    """Graph fields in DiffSpectra's format, in the coordinates' atom order, without stereochemistry."""
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    kekulized = Chem.Mol(mol)
    Chem.Kekulize(kekulized, clearAromaticFlags=True)
    n = mol.GetNumAtoms()
    rows, cols, types = [], [], []
    for bond in kekulized.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [i, j]
        cols += [j, i]
        types += [BOND_TYPES[bond.GetBondType()]] * 2
    edge_index = np.array([rows, cols], dtype=np.int64).reshape(2, -1)
    order = np.argsort(edge_index[0] * n + edge_index[1])
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return dict(
        atom_type=np.array([atom_types[z] for z in numbers], dtype=np.int64),
        charge=np.array(numbers, dtype=np.int64),
        fc=np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=np.int64),
        edge_index=edge_index[:, order],
        edge_type=np.array(types, dtype=np.int64)[order],
        pos=pos.astype(np.float32),
        num_atom=n,
        rdmol=mol,
    )


def _graphs_from_qm9(qm9_path, qm9_idx, molecules):
    qm9, slices = torch.load(qm9_path, weights_only=False)
    position = {int(idx): p for p, idx in enumerate(qm9.idx)}
    graphs = []
    for row, (numbers, pos) in enumerate(molecules):
        p = position.get(int(qm9_idx[row]) - 1)
        graph = None
        if p is not None:
            graph = _qm9_graph(qm9, slices, p)
        if graph is None or not _same_molecule(graph, numbers, pos):
            graphs.append(None)
            continue
        graph['pos'] = pos.astype(np.float32)
        graph['num_atom'] = len(numbers)
        graph['rdmol'] = _place(graph['rdmol'], pos)
        graph['bonds_from'] = 'qm9'
        graphs.append(graph)
    return graphs


def _qm9_graph(qm9, slices, p):
    graph = {}
    for key in ('atom_type', 'charge', 'fc', 'pos', 'edge_type'):
        graph[key] = getattr(qm9, key)[slices[key][p]:slices[key][p + 1]].numpy()
    graph['edge_index'] = qm9.edge_index[:, slices['edge_index'][p]:slices['edge_index'][p + 1]].numpy()
    graph['rdmol'] = qm9.rdmol[p]
    return graph


def _same_molecule(graph, numbers, pos):
    """Same elements in the same order, and every bond the same length within ``BOND_TOLERANCE``."""
    if not np.array_equal(graph['charge'], numbers):
        return False
    i, j = graph['edge_index']
    if len(i) == 0:
        return True
    lengths = np.linalg.norm(pos[i] - pos[j], axis=1)
    reference_lengths = np.linalg.norm(graph['pos'][i] - graph['pos'][j], axis=1)
    return np.abs(lengths - reference_lengths).max() <= BOND_TOLERANCE


def _place(mol, pos):
    """A sanitised copy of ``mol`` at the image's coordinates; unsanitised if RDKit rejects it."""
    placed = Chem.Mol(mol)
    conformer = placed.GetConformer()
    for k, (x, y, z) in enumerate(pos):
        conformer.SetAtomPosition(k, Point3D(float(x), float(y), float(z)))
    sanitised = Chem.Mol(placed)
    try:
        Chem.SanitizeMol(sanitised)
    except Exception:                           # a failed SanitizeMol leaves the Mol half-modified
        return placed
    return sanitised
