"""ETKDG conformer generation + MMFF optimization + Boltzmann weighting.

Kept deliberately dependency-light and import-safe (rdkit imported at call time)
so this module can be inspected/tested without rdkit installed.
"""
from __future__ import annotations

import numpy as np

KB_KCAL_PER_MOL_K = 0.0019872041  # Boltzmann constant in kcal/(mol*K)


def generate_conformers(
    smiles: str,
    n_confs: int = 8,
    prune_rms_thresh: float = 0.5,
    max_iters: int = 500,
    forcefield: str = "MMFF94",
    energy_window_kcal: float = 10.0,
    random_seed: int = 42,
) -> dict | None:
    """Generate conformers for one molecule, optimize, and compute Boltzmann weights.

    Returns a dict with atomic_nums, coords (n_kept, n_atoms, 3), energies_kcal
    (relative, min=0), boltzmann_weights (sum to 1), and which forcefield was
    actually used. Returns None if embedding fails outright (logged by caller).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = prune_rms_thresh
    params.numThreads = 0  # use all available cores

    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if len(cids) == 0:
        # fallback: some molecules (esp. macrocycles) need random coords to seed embedding
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if len(cids) == 0:
        return None

    used_ff = forcefield
    energies = []
    if forcefield == "MMFF94":
        mmff_props_ok = AllChem.MMFFHasAllMoleculeParams(mol)
        if mmff_props_ok:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=0)
            energies = [e for _converged, e in results]
        else:
            used_ff = "UFF"  # MMFF params missing for some atom type (e.g. certain metals/halogens)

    if used_ff == "UFF":
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=0)
        energies = [e for _converged, e in results]

    if not energies:
        return None

    energies = np.array(energies, dtype=np.float64)
    rel_energies = energies - energies.min()

    # drop conformers far above the minimum — negligible Boltzmann weight, just adds noise
    keep_mask = rel_energies <= energy_window_kcal
    kept_cids = [cid for cid, keep in zip(cids, keep_mask) if keep]
    rel_energies = rel_energies[keep_mask]

    if len(kept_cids) == 0:
        return None

    weights = np.exp(-rel_energies / (KB_KCAL_PER_MOL_K * 298.15))
    weights = weights / weights.sum()

    atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    coords = np.stack([mol.GetConformer(cid).GetPositions() for cid in kept_cids]).astype(np.float32)

    return {
        "atomic_nums": atomic_nums,
        "coords": coords,                        # (n_kept, n_atoms, 3)
        "energies_kcal": rel_energies.astype(np.float32),  # relative, min=0
        "boltzmann_weights": weights.astype(np.float32),   # sums to 1, at T=298.15K
        "forcefield_used": used_ff,
        "n_confs_kept": len(kept_cids),
    }


def boltzmann_weights_at_temperature(energies_kcal: np.ndarray, temperature_K: float) -> np.ndarray:
    """Recompute Boltzmann weights at an arbitrary temperature from cached relative energies.

    Useful for a temperature-sensitivity ablation later without regenerating conformers.
    """
    w = np.exp(-np.asarray(energies_kcal) / (KB_KCAL_PER_MOL_K * temperature_K))
    return w / w.sum()
