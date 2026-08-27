"""SMILES -> PyTorch Geometric graph (atom + bond features).

Feature choices are the standard MoleculeNet-style set: enough chemical
context (element, hybridization, aromaticity, ring membership, bond order,
conjugation, stereo) for a GNN to learn substructure patterns, without
pulling in anything exotic that would need extra justification.
"""
from __future__ import annotations

import torch

ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si"]  # + "other" bucket
DEGREES = [0, 1, 2, 3, 4, 5]              # + "other"
FORMAL_CHARGES = [-2, -1, 0, 1, 2]        # + "other"
NUM_HS = [0, 1, 2, 3, 4]                  # + "other"
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]  # + "other"
CHIRALITIES = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]  # + "other"

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]  # + "other"
STEREOS = ["STEREONONE", "STEREOZ", "STEREOE"]           # + "other"

ATOM_FEAT_DIM = (
    (len(ATOM_LIST) + 1) + (len(DEGREES) + 1) + (len(FORMAL_CHARGES) + 1)
    + (len(HYBRIDIZATIONS) + 1) + 1 + (len(NUM_HS) + 1) + 1 + (len(CHIRALITIES) + 1)
)
BOND_FEAT_DIM = (len(BOND_TYPES) + 1) + 1 + 1 + (len(STEREOS) + 1)


def _one_hot(value, choices) -> list[float]:
    """One-hot with an explicit trailing 'other' slot for unseen values."""
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def _atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_LIST)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + [1.0 if atom.IsInRing() else 0.0]
        + _one_hot(str(atom.GetChiralTag()), CHIRALITIES)
    )


def _bond_features(bond) -> list[float]:
    return (
        _one_hot(str(bond.GetBondType()), BOND_TYPES)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
        + _one_hot(str(bond.GetStereo()), STEREOS)
    )


def mol_to_graph_data(smiles: str, label: float | None = None):
    """Returns a torch_geometric.data.Data object, or None if the SMILES fails to parse."""
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float32)

    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        # add both directions — molecular graphs are undirected, message passing needs both
        edge_indices += [[i, j], [j, i]]
        edge_feats += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:  # single-atom molecule edge case, e.g. [Na+] fragments — no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.label = torch.tensor(label, dtype=torch.float32)  # named 'label' (not 'y') to match
    return data                                                  # the shared train_utils.py convention
