"""torch Dataset wrappers. One class per modality; multimodal notebooks combine them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from src.featurizers import encode
from src.graph_featurizer import mol_to_graph_data


class SelfiesDataset(Dataset):
    def __init__(self, selfies_list: list[str], labels: list[int], vocab: dict[str, int], max_len: int):
        self.encoded = [encode(s, vocab, max_len) for s in selfies_list]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids, true_len = self.encoded[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": true_len,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class ConformerDataset(Dataset):
    """Loads cached conformer .npz files (from notebook 02) for a list of molecule ids.

    Preloads everything into memory at init — the hERG-scale dataset is small
    enough (~600 molecules, <=8 conformers each) that this is faster than
    re-reading disk every epoch, without needing a fancier caching layer.
    """

    def __init__(self, mol_ids: list[str], labels: list[int], conformers_dir: str | Path):
        conformers_dir = Path(conformers_dir)
        self.items = []
        for mol_id, label in zip(mol_ids, labels):
            d = np.load(conformers_dir / f"{mol_id}.npz")
            self.items.append({
                "atomic_nums": torch.tensor(d["atomic_nums"], dtype=torch.long),
                "coords": torch.tensor(d["coords"], dtype=torch.float32),           # (n_conf, n_atoms, 3)
                "weights": torch.tensor(d["boltzmann_weights"], dtype=torch.float32),  # (n_conf,)
                "label": torch.tensor(label, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_conformers(batch_items: list[dict]) -> dict:
    """Flattens a list of per-molecule conformer sets into one batch with two
    levels of grouping: atom -> conformer (atom_conf_batch) and
    conformer -> molecule (conf_mol_batch). Needed because both the number of
    conformers per molecule and the number of atoms per molecule vary.
    """
    atom_z, atom_pos, atom_conf_batch = [], [], []
    conf_mol_batch, conf_weights, labels = [], [], []

    conf_counter = 0
    for mol_idx, item in enumerate(batch_items):
        n_conf = item["coords"].shape[0]
        for c in range(n_conf):
            atom_z.append(item["atomic_nums"])
            atom_pos.append(item["coords"][c])
            atom_conf_batch.append(torch.full((item["atomic_nums"].shape[0],), conf_counter, dtype=torch.long))
            conf_mol_batch.append(mol_idx)
            conf_weights.append(item["weights"][c])
            conf_counter += 1
        labels.append(item["label"])

    return {
        "atom_z": torch.cat(atom_z),
        "atom_pos": torch.cat(atom_pos, dim=0),
        "atom_conf_batch": torch.cat(atom_conf_batch),
        "conf_mol_batch": torch.tensor(conf_mol_batch, dtype=torch.long),
        "conf_weights": torch.tensor(conf_weights, dtype=torch.float32),
        "label": torch.stack(labels),
        "num_mols": len(batch_items),
    }


class MultimodalDataset(Dataset):
    """Combines SELFIES, graph, and conformer representations for the same molecules.

    Relies on ids/smiles/labels all being drawn from the same splits.json list
    (same order across every modality) so index i always refers to the same molecule.
    """

    def __init__(self, ids, smiles_list, selfies_list, labels, vocab, max_len, conformers_dir):
        self.selfies_encoded = [encode(s, vocab, max_len) for s in selfies_list]
        self.graphs = [mol_to_graph_data(smi) for smi in smiles_list]  # label=None, use top-level labels instead
        conformers_dir = Path(conformers_dir)
        self.conformers = []
        for mol_id in ids:
            d = np.load(conformers_dir / f"{mol_id}.npz")
            self.conformers.append({
                "atomic_nums": torch.tensor(d["atomic_nums"], dtype=torch.long),
                "coords": torch.tensor(d["coords"], dtype=torch.float32),
                "weights": torch.tensor(d["boltzmann_weights"], dtype=torch.float32),
            })
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids_, _true_len = self.selfies_encoded[idx]
        return {
            "input_ids": torch.tensor(ids_, dtype=torch.long),
            "graph": self.graphs[idx],
            "conformer": self.conformers[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def multimodal_collate(batch_items: list[dict]) -> dict:
    input_ids = torch.stack([item["input_ids"] for item in batch_items])
    graph_batch = Batch.from_data_list([item["graph"] for item in batch_items])

    # reuse collate_conformers for the 3D flattening logic, prefixing keys to avoid clashing
    conf_batch = collate_conformers([
        {**item["conformer"], "label": item["label"]} for item in batch_items
    ])

    labels = torch.stack([item["label"] for item in batch_items])

    return {
        "input_ids": input_ids,
        "graph": graph_batch,
        "conf_atom_z": conf_batch["atom_z"],
        "conf_atom_pos": conf_batch["atom_pos"],
        "conf_atom_conf_batch": conf_batch["atom_conf_batch"],
        "conf_conf_mol_batch": conf_batch["conf_mol_batch"],
        "conf_conf_weights": conf_batch["conf_weights"],
        "conf_num_mols": conf_batch["num_mols"],
        "label": labels,
    }

