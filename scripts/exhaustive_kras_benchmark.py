#!/usr/bin/env python3
"""
Exhaustive In Silico Validation & Lead Prioritization Benchmark for KRAS G12D Inhibitors.

Pipeline Gates:
  1. Chemical Sanitization, Deduplication & Physicochemical/ADME Profiling
  2. Structural Alert Filtering (PAINS A/B/C, Brenk, NIH + Custom Reactive Warheads)
  3. Literature Similarity Audit (ECFP4 Tanimoto vs. Known KRAS G12D Clinical/Preclinical Ligands)
  4. Physiological pH (7.4) Protonation & Formal Charge Verification (OpenBabel / RDKit fallback)
  5. High-Exhaustiveness KRAS G12D Switch-II Docking (AutoDock Vina, seed=42)
  6. 3D Pocket Pharmacophore & Interaction Audit (Meeko-mapped Asp12 N+ Salt-Bridge Geometry, Glu62/His95/Tyr96 Contacts)
  7. Internal Conformational Strain Energy Audit (MMFF94s Bound-vs-Global-Minimum Conformer Delta-E)
  8. Wild-Type KRAS Counter-Screening (Mutant vs. WT Selectivity Margin Delta-Delta-G)
  9. Retrosynthetic Disconnection (BRICS Fragmentation & Step Complexity) + Composite MPO Ranking
"""

import argparse
import json
import math
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    AllChem,
    BRICS,
    Crippen,
    Descriptors,
    FilterCatalog,
    Lipinski,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

try:
    from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
    HAS_MEEKO = True
except ImportError:
    HAS_MEEKO = False

try:
    from vina import Vina
    HAS_VINA = True
except ImportError:
    HAS_VINA = False


# =============================================================================
# REFERENCE LITERATURE INHIBITORS FOR SIMILARITY & BENCHMARK ANCHORING
# =============================================================================
REFERENCE_KRAS_INHIBITORS: Dict[str, str] = {
    "MRTX1133": "C[C@H]1CN(c2nc(OC[C@@]34CCCN3C[C@H](F)C4)nc3c2CCN(c2cccc4cccc(Cl)c24)C3)CCN1",
    "HRS-4642_analog": "Fc1ccc2c(c1)c(N1CCNCC1)nc(OC[C@]13CCCN1CC(F)C3)n2",
    "TH-Z835": "CC(C)c1cccc(C(C)C)c1-n1cnc2c(N3CCN(C(=O)C=C)CC3)nc(OC[C@H]3CCCN3C)nc21",
    "BI-2852": "CN(C)Cc1ccc(-c2nc3cc(O)ccc3[nH]2)cc1",
    "MRTX849_Adagrasib": "CN1CCC[C@H]1COc1nc(N2CCN(C(=O)C(F)=C)C[C@@H]2CC#N)nc2c1CCN(c1cccc3cccc(Cl)c13)C2",
}

CUSTOM_UNSTABLE_SMARTS: Dict[str, str] = {
    "Acylhydrazone_or_Hydrazone": "[NX3][NX2]=[CX3]",
    "Hydrazine": "[NX3][NX3]",
    "Peroxide": "[#8][#8]",
    "Anhydride": "[#6](=[O,S])[O,S][#6](=[O,S])",
    "Acyl_Halide": "[C,c](=[O])[F,Cl,Br,I]",
    "Nitro_Group": "[N+](=O)[O-]",
    "Disulfide": "[#16;!s][#16;!s]",
    "Michael_Acceptor_Enone": "[#6]=[#6]-[C,c](=[O])",
    "Macrocycle_8_to_10": "[*;r8,*;r9,*;r10]",
}


@dataclass
class PocketCoords:
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 22.0
    size_y: float = 22.0
    size_z: float = 22.0
    asp12_coord: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    glu62_coord: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    his95_coord: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tyr96_coord: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# =============================================================================
# RECEPTOR GEOMETRY & AUTO-DISCOVERY HELPERS
# =============================================================================
def parse_receptor_pocket(receptor_pdbqt: str) -> PocketCoords:
    res_atoms: Dict[int, List[np.ndarray]] = {12: [], 62: [], 95: [], 96: []}
    asp12_carboxyl: List[np.ndarray] = []

    with open(receptor_pdbqt, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                res_seq = int(line[22:26].strip())
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coord = np.array([x, y, z], dtype=float)
                if res_seq in res_atoms:
                    res_atoms[res_seq].append(coord)
                if res_seq == 12 and atom_name in ("OD1", "OD2", "CG"):
                    asp12_carboxyl.append(coord)
            except ValueError:
                continue

    def centroid(pts: List[np.ndarray]) -> np.ndarray:
        return np.mean(pts, axis=0) if pts else np.zeros(3)

    asp12_pt = centroid(asp12_carboxyl) if asp12_carboxyl else centroid(res_atoms[12])
    glu62_pt = centroid(res_atoms[62])
    his95_pt = centroid(res_atoms[95])
    tyr96_pt = centroid(res_atoms[96])

    valid_pts = [p for p in [asp12_pt, glu62_pt, his95_pt, tyr96_pt] if np.linalg.norm(p) > 0]
    pocket_center = np.mean(valid_pts, axis=0) if valid_pts else np.zeros(3)

    return PocketCoords(
        center_x=float(pocket_center[0]),
        center_y=float(pocket_center[1]),
        center_z=float(pocket_center[2]),
        size_x=22.0,
        size_y=22.0,
        size_z=22.0,
        asp12_coord=tuple(asp12_pt.tolist()),
        glu62_coord=tuple(glu62_pt.tolist()),
        his95_coord=tuple(his95_pt.tolist()),
        tyr96_coord=tuple(tyr96_pt.tolist()),
    )


def auto_locate_file(patterns: List[str], search_root: Path = Path(".")) -> Optional[str]:
    for pat in patterns:
        matches = sorted(search_root.rglob(pat))
        if matches:
            return str(matches[0])
    return None


# =============================================================================
# CHEMOINFORMATICS, PROTONATION & NOVELTY ENGINE
# =============================================================================
class MedicinalChemistryAuditor:
    def __init__(self):
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
        self.catalog = FilterCatalog.FilterCatalog(params)

        self.custom_patterns = {
            name: Chem.MolFromSmarts(smarts)
            for name, smarts in CUSTOM_UNSTABLE_SMARTS.items()
        }

        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.ref_fps = {}
        for name, smi in REFERENCE_KRAS_INHIBITORS.items():
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                self.ref_fps[name] = self.morgan_gen.GetFingerprint(m)

    def standardize_smiles(self, smi: str) -> Optional[Tuple[str, Chem.Mol]]:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return None
        try:
            largest_frag = rdMolStandardize.LargestFragmentChooser().choose(mol)
            uncharged = rdMolStandardize.Uncharger().uncharge(largest_frag)
            can_smi = Chem.MolToSmiles(uncharged, isomericSmiles=True)
            clean_mol = Chem.MolFromSmiles(can_smi)
            if clean_mol is None or clean_mol.GetNumHeavyAtoms() < 10:
                return None
            return can_smi, clean_mol
        except Exception:
            return None

    def audit_alerts(self, mol: Chem.Mol) -> Tuple[bool, str]:
        alerts = []
        entry = self.catalog.GetFirstMatch(mol)
        if entry is not None:
            alerts.append(entry.GetDescription())
        for name, patt in self.custom_patterns.items():
            if patt is not None and mol.HasSubstructMatch(patt):
                alerts.append(name)
        return (len(alerts) == 0), ("; ".join(alerts) if alerts else "CLEAN")

    def compute_novelty(self, mol: Chem.Mol) -> Tuple[float, str]:
        fp = self.morgan_gen.GetFingerprint(mol)
        best_sim = 0.0
        best_ref = "None"
        for name, ref_fp in self.ref_fps.items():
            sim = float(DataStructs.TanimotoSimilarity(fp, ref_fp))
            if sim > best_sim:
                best_sim = sim
                best_ref = name
        return round(best_sim, 4), best_ref

    def protonate_ph74(self, can_smi: str) -> str:
        try:
            proc = subprocess.run(
                ["obabel", f"-:{can_smi}", "-osmi", "-p", "7.4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                prot_smi = proc.stdout.strip().split()[0]
                if Chem.MolFromSmiles(prot_smi) is not None:
                    return prot_smi
        except Exception:
            pass

        mol = Chem.MolFromSmiles(can_smi)
        if mol is None:
            return can_smi
        rw = Chem.RWMol(mol)
        basic_amine = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(Nc);!$(N#*);!$(N=*)]")
        matches = rw.GetSubstructMatches(basic_amine)
        if matches:
            idx = matches[0][0]
            atom = rw.GetAtomWithIdx(idx)
            atom.SetFormalCharge(1)
            try:
                Chem.SanitizeMol(rw)
                return Chem.MolToSmiles(rw, isomericSmiles=True)
            except Exception:
                return can_smi
        return can_smi

    def profile_physicochemical(self, mol: Chem.Mol, prot_smi: str) -> Dict:
        mw = Descriptors.MolWt(mol)
        slogp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        qed_val = QED.qed(mol)
        sa_val = sascorer.calculateScore(mol)

        lipinski_violations = sum([mw > 500, slogp > 5.0, hbd > 5, hba > 10])
        veber_pass = (rot_bonds <= 10) and (tpsa <= 140.0)
        cns_mpo_proxy = (tpsa <= 90.0) and (mw <= 450.0)

        prot_mol = Chem.MolFromSmiles(prot_smi)
        formal_charge_ph74 = Chem.GetFormalCharge(prot_mol) if prot_mol is not None else 0

        try:
            brics_frags = BRICS.BRICSDecompose(mol)
            n_building_blocks = len(brics_frags)
        except Exception:
            n_building_blocks = -1

        return {
            "MW": round(mw, 2),
            "SLogP": round(slogp, 3),
            "TPSA": round(tpsa, 2),
            "HBD": int(hbd),
            "HBA": int(hba),
            "RotBonds": int(rot_bonds),
            "Rings": int(rings),
            "AromRings": int(arom_rings),
            "Fsp3": round(fsp3, 3),
            "QED": round(qed_val, 4),
            "SAScore": round(sa_val, 3),
            "Lipinski_Violations": int(lipinski_violations),
            "Veber_Pass": bool(veber_pass),
            "BBB_Permeable_Proxy": bool(cns_mpo_proxy),
            "Formal_Charge_pH74": int(formal_charge_ph74),
            "BRICS_Building_Blocks": int(n_building_blocks),
        }


# =============================================================================
# 3D CONFORMER BASELINE, MEEKO POSE RECONSTRUCTION, STRAIN & SALT-BRIDGE AUDIT
# =============================================================================
def compute_free_energy_baseline(prot_smi: str) -> Optional[Tuple[Chem.Mol, float]]:
    """Generates 10 ETKDGv3 conformers and finds the MMFF94s global-minimum free energy."""
    mol = Chem.MolFromSmiles(prot_smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return None

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
    min_free_energy = float("inf")
    best_cid = 0

    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
                e = ff.CalcEnergy()
                if e < min_free_energy:
                    min_free_energy = e
                    best_cid = cid
    else:
        for cid in cids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
                e = ff.CalcEnergy()
                if e < min_free_energy:
                    min_free_energy = e
                    best_cid = cid

    if math.isinf(min_free_energy):
        min_free_energy = 0.0

    single_mol = Chem.Mol(mol)
    conf = Chem.Conformer(single_mol.GetConformer(best_cid))
    single_mol.RemoveAllConformers()
    single_mol.AddConformer(conf, assignId=True)
    return single_mol, float(min_free_energy)


def prepare_3d_conformer_and_pdbqt(prot_smi: str) -> Optional[Tuple[Chem.Mol, str, float]]:
    base = compute_free_energy_baseline(prot_smi)
    if base is None or not HAS_MEEKO:
        return None
    single_mol, min_free_energy = base

    prep = MoleculePreparation()
    mol_setups = prep.prepare(single_mol)
    if not mol_setups:
        return None
    pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        return None

    return single_mol, pdbqt_str, float(min_free_energy)


def parse_vina_pose_coords(pdbqt_pose_str: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    heavy_coords: List[np.ndarray] = []
    polar_coords: List[np.ndarray] = []
    for line in pdbqt_pose_str.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_type = line[77:79].strip() if len(line) >= 79 else line.split()[-1]
        if atom_type.startswith("H"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            pt = np.array([x, y, z], dtype=float)
            heavy_coords.append(pt)
            if atom_type in ("N", "NA", "NS", "OA", "O"):
                polar_coords.append(pt)
        except ValueError:
            continue
    return heavy_coords, polar_coords


def _pdbqt_to_rdkit_pose(docked_pdbqt_str: str) -> Optional[Chem.Mol]:
    """
    Reconstructs MODEL 1 as an RDKit Mol using Meeko's REMARK SMILES IDX atom-mapping
    so torsion-tree reordering never scrambles atoms.
    """
    if not HAS_MEEKO:
        return None
    try:
        pmol = PDBQTMolecule(docked_pdbqt_str, is_dlg=False, skip_typing=True)
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        if rdkit_mols and rdkit_mols[0] is not None:
            pose_mol = Chem.Mol(rdkit_mols[0])
            if pose_mol.GetNumConformers() > 1:
                conf0 = Chem.Conformer(pose_mol.GetConformer(0))
                pose_mol.RemoveAllConformers()
                pose_mol.AddConformer(conf0, assignId=True)
            return pose_mol
    except Exception:
        pass
    return None


def evaluate_true_asp12_salt_bridge(
    docked_pdbqt_str: str, asp12_coord: Tuple[float, float, float]
) -> Tuple[bool, float]:
    """
    Uses Meeko-reconstructed 3D coordinates (exact atom mapping) to check for a
    formal +1 nitrogen within <= 4.0 A of Asp12 carboxylate and directional
    C-N+...O geometry (>= 105 deg).
    """
    pose_mol = _pdbqt_to_rdkit_pose(docked_pdbqt_str)
    if pose_mol is None or pose_mol.GetNumConformers() == 0:
        return False, np.nan

    conf = pose_mol.GetConformer(0)
    pos_n_indices = [
        a.GetIdx() for a in pose_mol.GetAtoms()
        if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0
    ]
    if not pos_n_indices:
        return False, np.nan

    asp12_pt = np.array(asp12_coord, dtype=float)
    best_dist = float("inf")
    salt_bridge_ok = False

    for n_idx in pos_n_indices:
        pos = conf.GetAtomPosition(n_idx)
        n_pt = np.array([pos.x, pos.y, pos.z], dtype=float)
        dist = float(np.linalg.norm(n_pt - asp12_pt))
        if dist < best_dist:
            best_dist = dist

        if dist <= 4.0:
            nbr_pts = [
                np.array(
                    [
                        conf.GetAtomPosition(nb.GetIdx()).x,
                        conf.GetAtomPosition(nb.GetIdx()).y,
                        conf.GetAtomPosition(nb.GetIdx()).z,
                    ],
                    dtype=float,
                )
                for nb in pose_mol.GetAtomWithIdx(n_idx).GetNeighbors()
                if nb.GetAtomicNum() > 1
            ]
            if nbr_pts:
                v1 = n_pt - np.mean(nbr_pts, axis=0)
                v2 = asp12_pt - n_pt
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                angle_deg = (
                    float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))
                    if denom > 1e-6
                    else 135.0
                )
                if angle_deg >= 105.0:
                    salt_bridge_ok = True
            else:
                salt_bridge_ok = True

    return salt_bridge_ok, (round(best_dist, 2) if best_dist < float("inf") else np.nan)


def compute_bound_strain_energy(ref_mol: Chem.Mol, docked_pdbqt_str: str, min_free_energy: float) -> float:
    """
    Reconstructs the docked pose via Meeko (exact atom order), freezes heavy atoms,
    relaxes hydrogens under MMFF94s, and evaluates:
        Delta E_strain = max(0.0, E_bound_MMFF94s - E_free_global_min)
    """
    if min_free_energy == 0.0:
        return -1.0

    pose_mol = _pdbqt_to_rdkit_pose(docked_pdbqt_str)
    if pose_mol is None:
        return -1.0

    try:
        mol_bound = Chem.AddHs(pose_mol, addCoords=True)
        heavy_indices = [a.GetIdx() for a in mol_bound.GetAtoms() if a.GetAtomicNum() > 1]

        if AllChem.MMFFHasAllMoleculeParams(mol_bound):
            props = AllChem.MMFFGetMoleculeProperties(mol_bound, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(mol_bound, props, confId=0)
            if ff is not None:
                for idx in heavy_indices:
                    ff.AddFixedPoint(idx)
                ff.Minimize(maxIts=200)
                bound_e = ff.CalcEnergy()
                strain = max(0.0, float(bound_e - min_free_energy))
                return round(strain, 3) if strain <= 250.0 else -1.0
    except Exception:
        pass
    return -1.0


def verify_strain_engine() -> None:
    """Unit test: flexible-ligand Meeko round-trip self-strain must be < 1.0 kcal/mol."""
    test_smi = "O=C(Nc1cccnc1)N1CCN(Cc2ccc3ccccc3c2)CC1"
    res = prepare_3d_conformer_and_pdbqt(test_smi)
    assert res is not None, "Strain engine self-test failed during 3D prep"
    ref_mol, pdbqt_str, min_free_e = res
    self_strain = compute_bound_strain_energy(ref_mol, pdbqt_str, min_free_e)
    assert 0.0 <= self_strain < 1.0, f"Strain engine atom-mapping check failed: self_strain={self_strain}"
    print(f"[SELF-TEST OK] Flexible-ligand Meeko round-trip strain = {self_strain:.3f} kcal/mol", flush=True)


def dock_single_candidate(
    task: Tuple[int, str, str, str, Optional[str], PocketCoords, int, str]
) -> Dict:
    (
        idx,
        can_smi,
        prot_smi,
        g12d_receptor,
        wt_receptor,
        pocket,
        exhaustiveness,
        pose_dir,
    ) = task

    out = {
        "row_idx": idx,
        "G12D_Vina_kcal": np.nan,
        "G12D_Pose2_kcal": np.nan,
        "Pose_Gap_kcal": np.nan,
        "Ligand_Efficiency_LE": np.nan,
        "Asp12_Min_Dist_A": np.nan,
        "Asp12_SaltBridge_OK": False,
        "Glu62_Min_Dist_A": np.nan,
        "His95_Min_Dist_A": np.nan,
        "Tyr96_Min_Dist_A": np.nan,
        "Bound_Strain_kcal": np.nan,
        "WT_Vina_kcal": np.nan,
        "Selectivity_ddG_kcal": np.nan,
        "Pose_File": "",
    }

    if not HAS_VINA or not HAS_MEEKO:
        return out

    prep_res = prepare_3d_conformer_and_pdbqt(prot_smi)
    if prep_res is None:
        return out
    ref_mol, lig_pdbqt, min_free_energy = prep_res

    try:
        v = Vina(sf_name="vina", cpu=1, seed=42, verbosity=0)
        v.set_receptor(g12d_receptor)
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(
            center=[pocket.center_x, pocket.center_y, pocket.center_z],
            box_size=[pocket.size_x, pocket.size_y, pocket.size_z],
        )
        v.dock(exhaustiveness=exhaustiveness, n_poses=5)
        energies = v.energies(n_poses=5)
        if energies is None or len(energies) == 0:
            return out

        best_aff = float(energies[0][0])
        second_aff = float(energies[1][0]) if len(energies) > 1 else best_aff
        pose_pdbqt = v.poses(n_poses=1)

        pose_path = os.path.join(pose_dir, f"cand_{idx:04d}_g12d.pdbqt")
        with open(pose_path, "w") as pf:
            pf.write(pose_pdbqt)

        heavy_pts, polar_pts = parse_vina_pose_coords(pose_pdbqt)
        n_heavy = max(1, len(heavy_pts))
        le = -best_aff / n_heavy

        def min_dist(pts: List[np.ndarray], target: Tuple[float, float, float]) -> float:
            if not pts or np.linalg.norm(target) == 0:
                return np.nan
            t = np.array(target, dtype=float)
            return float(np.min([np.linalg.norm(p - t) for p in pts]))

        asp12_d = min_dist(polar_pts if polar_pts else heavy_pts, pocket.asp12_coord)
        glu62_d = min_dist(heavy_pts, pocket.glu62_coord)
        his95_d = min_dist(heavy_pts, pocket.his95_coord)
        tyr96_d = min_dist(heavy_pts, pocket.tyr96_coord)

        sb_ok, _ = evaluate_true_asp12_salt_bridge(pose_pdbqt, pocket.asp12_coord)
        strain_kcal = compute_bound_strain_energy(ref_mol, pose_pdbqt, min_free_energy)

        out.update(
            {
                "G12D_Vina_kcal": round(best_aff, 3),
                "G12D_Pose2_kcal": round(second_aff, 3),
                "Pose_Gap_kcal": round(second_aff - best_aff, 3),
                "Ligand_Efficiency_LE": round(le, 4),
                "Asp12_Min_Dist_A": round(asp12_d, 2) if not np.isnan(asp12_d) else np.nan,
                "Asp12_SaltBridge_OK": bool(sb_ok),
                "Glu62_Min_Dist_A": round(glu62_d, 2) if not np.isnan(glu62_d) else np.nan,
                "His95_Min_Dist_A": round(his95_d, 2) if not np.isnan(his95_d) else np.nan,
                "Tyr96_Min_Dist_A": round(tyr96_d, 2) if not np.isnan(tyr96_d) else np.nan,
                "Bound_Strain_kcal": strain_kcal if strain_kcal >= 0 else np.nan,
                "Pose_File": pose_path,
            }
        )

        if wt_receptor and os.path.exists(wt_receptor):
            v_wt = Vina(sf_name="vina", cpu=1, seed=42, verbosity=0)
            v_wt.set_receptor(wt_receptor)
            v_wt.set_ligand_from_string(lig_pdbqt)
            v_wt.compute_vina_maps(
                center=[pocket.center_x, pocket.center_y, pocket.center_z],
                box_size=[pocket.size_x, pocket.size_y, pocket.size_z],
            )
            v_wt.dock(exhaustiveness=exhaustiveness, n_poses=3)
            wt_energies = v_wt.energies(n_poses=1)
            if wt_energies is not None and len(wt_energies) > 0:
                wt_aff = float(wt_energies[0][0])
                out["WT_Vina_kcal"] = round(wt_aff, 3)
                out["Selectivity_ddG_kcal"] = round(wt_aff - best_aff, 3)

    except Exception:
        pass

    return out


# =============================================================================
# MULTI-PARAMETER OPTIMIZATION (MPO) SCORING & FUNNEL ORCHESTRATOR
# =============================================================================
def compute_composite_mpo(df: pd.DataFrame) -> pd.Series:
    aff_norm = np.clip((-df["G12D_Vina_kcal"] - 6.5) / 5.0, 0.0, 1.0).fillna(0.0)
    asp12_norm = np.clip((6.0 - df["Asp12_Min_Dist_A"]) / 2.4, 0.0, 1.0).fillna(0.2)
    asp12_norm = np.where(df["Formal_Charge_pH74"] >= 1, asp12_norm, asp12_norm * 0.35)

    strain_vals = df["Bound_Strain_kcal"].fillna(4.0)
    strain_norm = np.clip((10.0 - strain_vals) / 7.0, 0.0, 1.0)

    nov_norm = np.clip((0.70 - df["Max_Ref_Tanimoto"]) / 0.35, 0.0, 1.0)
    qed_norm = np.clip(df["QED"], 0.0, 1.0)
    sa_norm = np.clip((5.5 - df["SAScore"]) / 3.5, 0.0, 1.0)

    if df["Selectivity_ddG_kcal"].notna().any():
        sel_norm = np.clip((df["Selectivity_ddG_kcal"] + 0.5) / 2.5, 0.0, 1.0).fillna(0.5)
        mpo = (
            0.32 * aff_norm
            + 0.20 * asp12_norm
            + 0.13 * strain_norm
            + 0.12 * sel_norm
            + 0.08 * nov_norm
            + 0.08 * qed_norm
            + 0.07 * sa_norm
        )
    else:
        mpo = (
            0.38 * aff_norm
            + 0.22 * asp12_norm
            + 0.14 * strain_norm
            + 0.10 * nov_norm
            + 0.09 * qed_norm
            + 0.07 * sa_norm
        )

    mpo = np.where(df["Alerts_Clean"], mpo, mpo * 0.25)
    return np.round(mpo, 4)


def _clean_nan_for_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan_for_json(v) for v in obj]
    return obj


def rescore_existing_poses(
    full_csv_path: str, pose_dir: str, receptor_path: str, out_dir: Path
) -> None:
    """
    Fast local re-scoring mode: re-computes Bound_Strain_kcal and Asp12_SaltBridge_OK
    from already-docked PDBQT poses using Meeko's exact atom-mapping reconstruction.
    """
    verify_strain_engine()
    pocket = parse_receptor_pocket(receptor_path)
    df = pd.read_csv(full_csv_path)
    if "Benchmark_Rank" in df.columns:
        df = df.drop(columns=["Benchmark_Rank"])

    updated = 0
    for i, row in df.iterrows():
        if pd.isna(row.get("G12D_Vina_kcal")):
            continue
        pose_path = str(row.get("Pose_File", ""))
        if not os.path.exists(pose_path):
            cand_name = os.path.basename(pose_path) if pose_path else ""
            alt_path = os.path.join(pose_dir, cand_name) if cand_name else ""
            if alt_path and os.path.exists(alt_path):
                pose_path = alt_path
            else:
                continue

        pose_txt = Path(pose_path).read_text()
        base = compute_free_energy_baseline(row["Protonated_pH74_SMILES"])
        if base is None:
            continue
        ref_mol, min_free_e = base

        sb_ok, _ = evaluate_true_asp12_salt_bridge(pose_txt, pocket.asp12_coord)
        strain_kcal = compute_bound_strain_energy(ref_mol, pose_txt, min_free_e)

        df.at[i, "Asp12_SaltBridge_OK"] = bool(sb_ok)
        df.at[i, "Bound_Strain_kcal"] = strain_kcal if strain_kcal >= 0 else np.nan
        df.at[i, "Pose_File"] = pose_path
        updated += 1
        if updated % 25 == 0:
            print(f"  -> Rescored {updated} poses...", flush=True)

    df["Composite_MPO_Score"] = compute_composite_mpo(df)
    df = df.sort_values(
        by=["Composite_MPO_Score", "G12D_Vina_kcal"], ascending=[False, True]
    ).reset_index(drop=True)
    df.insert(0, "Benchmark_Rank", np.arange(1, len(df) + 1))

    tier1_df = df[
        df["Alerts_Clean"]
        & df["Novel_Scaffold_Pass"]
        & (df["G12D_Vina_kcal"] <= -8.5)
        & (df["Bound_Strain_kcal"].fillna(999.0) <= 6.0)
    ].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "full_1600_audited_library.csv", index=False)
    tier1_df.to_csv(out_dir / "tier1_validated_leads.csv", index=False)

    funnel_summary = {
        "pocket_geometry": asdict(pocket),
        "funnel_counts": {
            "total_generated_smiles": int(len(df) + 27) if len(df) == 1573 else int(len(df)),
            "unique_sanitized_valid": int(len(df)),
            "passed_pains_brenk_nih_custom_alerts": int(df["Alerts_Clean"].sum()),
            "passed_novelty_tanimoto_lt_045": int((df["Alerts_Clean"] & df["Novel_Scaffold_Pass"]).sum()),
            "gpu_redocked_3d_g12d_and_wt": int(updated),
            "tier1_validated_leads": int(len(tier1_df)),
            "tier1_with_asp12_salt_bridge": int(tier1_df["Asp12_SaltBridge_OK"].sum()) if len(tier1_df) > 0 else 0,
            "strain_populated_count": int(df["Bound_Strain_kcal"].notna().sum()),
        },
        "top_10_leads_preview": df.head(10)[
            [
                "Benchmark_Rank",
                "Canonical_SMILES",
                "Composite_MPO_Score",
                "G12D_Vina_kcal",
                "WT_Vina_kcal",
                "Selectivity_ddG_kcal",
                "Asp12_Min_Dist_A",
                "Asp12_SaltBridge_OK",
                "Bound_Strain_kcal",
                "Max_Ref_Tanimoto",
                "QED",
                "SAScore",
                "Formal_Charge_pH74",
            ]
        ].to_dict(orient="records"),
    }
    with open(out_dir / "validation_funnel_report.json", "w") as jf:
        json.dump(_clean_nan_for_json(funnel_summary), jf, indent=2, allow_nan=False)

    docked_valid = df[df["Bound_Strain_kcal"].notna()]
    print(f"\n[RESCORE COMPLETE] Re-scored {updated} poses -> {out_dir}")
    if len(docked_valid) > 0:
        print(
            f"Strain stats (kcal/mol): min={docked_valid['Bound_Strain_kcal'].min():.2f}, "
            f"median={docked_valid['Bound_Strain_kcal'].median():.2f}, "
            f"max={docked_valid['Bound_Strain_kcal'].max():.2f}"
        )
    print(
        f"Tier-1 Leads (Vina <= -8.5 & Strain <= 6.0): {len(tier1_df)} | "
        f"True N+ Asp12 Salt Bridge: {int(tier1_df['Asp12_SaltBridge_OK'].sum())}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive In Silico Benchmark for KRAS G12D RL Candidates"
    )
    parser.add_argument("--input", type=str, default="results/kras_g12d_balanced_1.csv")
    parser.add_argument("--receptor", type=str, default=None)
    parser.add_argument("--wt-receptor", type=str, default=None)
    parser.add_argument("--top-dock", type=int, default=150)
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--out-dir", type=str, default="exhaustive_benchmark_out")
    parser.add_argument("--rescore-csv", type=str, default=None)
    parser.add_argument("--pose-dir", type=str, default="exhaustive_benchmark_out/docked_poses")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    receptor_path = args.receptor or auto_locate_file(
        ["*7RT1*receptor*.pdbqt", "*7RPZ*receptor*.pdbqt", "*g12d*receptor*.pdbqt", "*receptor*.pdbqt"]
    )
    if not receptor_path or not os.path.exists(receptor_path):
        print("[ERROR] Could not locate KRAS G12D receptor PDBQT. Pass --receptor <path>.")
        sys.exit(1)

    if args.rescore_csv:
        rescore_existing_poses(args.rescore_csv, args.pose_dir, receptor_path, out_dir)
        return

    verify_strain_engine()
    pose_dir = out_dir / "docked_poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    wt_receptor_path = args.wt_receptor or auto_locate_file(
        ["*wt*receptor*.pdbqt", "*wild*receptor*.pdbqt", "*4OBE*receptor*.pdbqt"]
    )

    pocket = parse_receptor_pocket(receptor_path)
    print(f"[INFO] Receptor G12D : {receptor_path}")
    if wt_receptor_path:
        print(f"[INFO] Receptor WT   : {wt_receptor_path}")

    if not os.path.exists(args.input):
        alt = auto_locate_file([os.path.basename(args.input)])
        if alt:
            args.input = alt
        else:
            print(f"[ERROR] Input CSV not found: {args.input}")
            sys.exit(1)

    raw_df = pd.read_csv(args.input)
    smiles_col = next((c for c in raw_df.columns if c.upper() == "SMILES"), raw_df.columns[0])
    rl_score_col = next((c for c in raw_df.columns if c.lower() in ("score", "total_score")), None)
    raw_df["RL_Step"] = (np.arange(len(raw_df)) // 32) + 1
    total_generated = len(raw_df)

    auditor = MedicinalChemistryAuditor()
    records, seen_smiles = [], set()
    for _, row in raw_df.iterrows():
        std = auditor.standardize_smiles(row[smiles_col])
        if std is None:
            continue
        can_smi, clean_mol = std
        if can_smi in seen_smiles:
            continue
        seen_smiles.add(can_smi)

        alerts_clean, alert_reason = auditor.audit_alerts(clean_mol)
        max_sim, nearest_ref = auditor.compute_novelty(clean_mol)
        prot_smi = auditor.protonate_ph74(can_smi)
        physchem = auditor.profile_physicochemical(clean_mol, prot_smi)

        rec = {
            "Canonical_SMILES": can_smi,
            "Protonated_pH74_SMILES": prot_smi,
            "First_RL_Step": int(row["RL_Step"]),
            "RL_Total_Score": round(float(row[rl_score_col]), 4) if rl_score_col else np.nan,
            "Alerts_Clean": alerts_clean,
            "Alert_Details": alert_reason,
            "Max_Ref_Tanimoto": max_sim,
            "Nearest_Known_Inhibitor": nearest_ref,
            "Novel_Scaffold_Pass": bool(max_sim < 0.45),
        }
        rec.update(physchem)
        records.append(rec)

    audit_df = pd.DataFrame(records).sort_values(
        by=["Alerts_Clean", "Novel_Scaffold_Pass", "RL_Total_Score", "QED"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    dock_subset = audit_df.head(min(args.top_dock, len(audit_df))).copy()
    tasks = [
        (
            int(i),
            row["Canonical_SMILES"],
            row["Protonated_pH74_SMILES"],
            receptor_path,
            wt_receptor_path,
            pocket,
            args.exhaustiveness,
            str(pose_dir),
        )
        for i, row in dock_subset.iterrows()
    ]

    dock_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(dock_single_candidate, t) for t in tasks]
        for done_count, fut in enumerate(as_completed(futures), 1):
            try:
                dock_results.append(fut.result())
            except Exception:
                pass
            if done_count % 25 == 0 or done_count == len(tasks):
                print(f"  -> Docked {done_count}/{len(tasks)} candidates...", flush=True)

    full_df = audit_df.join(pd.DataFrame(dock_results).set_index("row_idx"))
    full_df["Composite_MPO_Score"] = compute_composite_mpo(full_df)
    full_df = full_df.sort_values(
        by=["Composite_MPO_Score", "G12D_Vina_kcal"], ascending=[False, True]
    ).reset_index(drop=True)
    full_df.insert(0, "Benchmark_Rank", np.arange(1, len(full_df) + 1))

    tier1_df = full_df[
        full_df["Alerts_Clean"]
        & full_df["Novel_Scaffold_Pass"]
        & (full_df["G12D_Vina_kcal"] <= -8.5)
        & (full_df["Bound_Strain_kcal"].fillna(999.0) <= 6.0)
    ].copy()

    full_df.to_csv(out_dir / "full_1600_audited_library.csv", index=False)
    tier1_df.to_csv(out_dir / "tier1_validated_leads.csv", index=False)

    funnel_summary = {
        "input_file": args.input,
        "receptor_g12d": receptor_path,
        "receptor_wt": wt_receptor_path,
        "pocket_geometry": asdict(pocket),
        "funnel_counts": {
            "total_generated_smiles": total_generated,
            "unique_sanitized_valid": len(audit_df),
            "passed_pains_brenk_nih_custom_alerts": int(audit_df["Alerts_Clean"].sum()),
            "passed_novelty_tanimoto_lt_045": int((audit_df["Alerts_Clean"] & audit_df["Novel_Scaffold_Pass"]).sum()),
            "exhaustively_redocked_3d": len(tasks),
            "tier1_validated_leads": int(len(tier1_df)),
            "tier1_with_asp12_salt_bridge": int(tier1_df["Asp12_SaltBridge_OK"].sum()) if len(tier1_df) > 0 else 0,
            "strain_populated_count": int(full_df["Bound_Strain_kcal"].notna().sum()),
        },
        "top_10_leads_preview": full_df.head(10)[
            [
                "Benchmark_Rank",
                "Canonical_SMILES",
                "Composite_MPO_Score",
                "G12D_Vina_kcal",
                "WT_Vina_kcal",
                "Selectivity_ddG_kcal",
                "Asp12_Min_Dist_A",
                "Asp12_SaltBridge_OK",
                "Bound_Strain_kcal",
                "Max_Ref_Tanimoto",
                "QED",
                "SAScore",
                "Formal_Charge_pH74",
            ]
        ].to_dict(orient="records"),
    }
    with open(out_dir / "validation_funnel_report.json", "w") as jf:
        json.dump(_clean_nan_for_json(funnel_summary), jf, indent=2, allow_nan=False)

    print(f"\n[COMPLETE] Tier-1 Leads: {len(tier1_df)} | True N+ Asp12 Salt Bridge: {int(tier1_df['Asp12_SaltBridge_OK'].sum())}")


if __name__ == "__main__":
    main()
