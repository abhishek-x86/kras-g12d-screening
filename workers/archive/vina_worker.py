import sys
import os
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

RECEPTOR = "/home/abhi/kras_screening/7RT1_fixed_receptor.pdbqt"
CENTER = [-1.810, -1.541, -25.541]
BOX_SIZE = [18.0, 18.0, 18.0]

# Suppress low-level C stdout/stderr during worker initialization and runs
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
            return 0.0

        prepper = MoleculePreparation()
        mol_setups = prepper.prepare(mol_h)
        lig_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

        # Mute Vina runtime logs during docking
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            v_engine.set_ligand_from_string(lig_str)
            v_engine.dock(exhaustiveness=2, n_poses=1)
            aff = float(v_engine.energies(n_poses=1)[0][0])
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            os.close(devnull)

        # -3.0 kcal/mol -> 0.0, -9.0 kcal/mol -> 1.0
        reward = (-aff - 3.0) / 6.0
        return float(np.clip(reward, 0.0, 1.0))
    except Exception:
        return 0.0

def main():
    raw_input = sys.stdin.read().strip()
    if not raw_input:
        sys.exit(0)

    smiles_list = [line.strip() for line in raw_input.splitlines() if line.strip()]
    scores = [evaluate_mol(s) for s in smiles_list]

    output = {
        "payload": {
            "docking_score": scores
        }
    }
    sys.stdout.write(json.dumps(output))

if __name__ == "__main__":
    main()
