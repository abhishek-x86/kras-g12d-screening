import sys
import os
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

BASE_DIR = Path(__file__).resolve().parent
RECEPTOR = str(BASE_DIR / "7RT1_fixed_receptor.pdbqt")
CENTER = [-1.810, -1.541, -25.541]
BOX_SIZE = [18.0, 18.0, 18.0]

ASP12_OXYGENS = []
with open(RECEPTOR) as f:
    for line in f:
        if line.startswith("ATOM") and line[17:20].strip() == "ASP" and line[22:26].strip() == "12":
            if line[12:16].strip() in {"OD1", "OD2"}:
                ASP12_OXYGENS.append(np.array([
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ]))

def init_vina():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        v = Vina(sf_name="vina", cpu=4, verbosity=0)
        v.set_receptor(RECEPTOR)
        v.compute_vina_maps(center=CENTER, box_size=BOX_SIZE)
        return v
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)

v_engine = init_vina()

def min_asp12_polar_dist(pdbqt_pose):
    min_d = 99.0
    for line in pdbqt_pose.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            elem = line[76:78].strip() or line[12:16].strip()[0]
            if elem in {"N", "O", "S"}:
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                for asp_xyz in ASP12_OXYGENS:
                    d = float(np.linalg.norm(xyz - asp_xyz))
                    if d < min_d:
                        min_d = d
    return min_d

def evaluate_mol(smi):
    try:
        smi = smi.strip()
        if not smi:
            return 0.0

        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return 0.0

        mw = Descriptors.MolWt(mol)
        if mw < 200 or mw > 750:
            return 0.0

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol_h, params) != 0:
                return 0.0

        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)

        conf = mol_h.GetConformer()
        centroid = AllChem.ComputeCentroid(conf)
        pocket = Point3D(*CENTER)
        for idx in range(mol_h.GetNumAtoms()):
            p = conf.GetAtomPosition(idx)
            conf.SetAtomPosition(idx, Point3D(
                p.x - centroid.x + pocket.x,
                p.y - centroid.y + pocket.y,
                p.z - centroid.z + pocket.z
            ))

        prepper = MoleculePreparation()
        lig_str = PDBQTWriterLegacy.write_string(prepper.prepare(mol_h)[0])[0]

        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            v_engine.set_ligand_from_string(lig_str)
            v_engine.dock(exhaustiveness=4, n_poses=1)
            aff = float(v_engine.energies(n_poses=1)[0][0])
            pose_str = v_engine.poses(n_poses=1)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            os.close(devnull)

        base_reward = np.clip((-aff - 3.0) / 6.0, 0.0, 1.0)
        asp_dist = min_asp12_polar_dist(pose_str)
        selectivity_bonus = 1.0 if asp_dist <= 3.3 else 0.75
        return float(np.clip(base_reward * selectivity_bonus, 0.0, 1.0))
    except Exception:
        return 0.0

def main():
    raw_input = sys.stdin.read().strip()
    if not raw_input:
        sys.exit(0)

    smiles_list = [line.strip() for line in raw_input.splitlines() if line.strip()]
    scores = [evaluate_mol(s) for s in smiles_list]
    sys.stdout.write(json.dumps({"payload": {"docking_score": scores}}))

if __name__ == "__main__":
    main()
