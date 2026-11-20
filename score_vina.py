import sys
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

RECEPTOR = "7RT1_fixed_receptor.pdbqt"
CENTER = [-1.810, -1.541, -25.541]
BOX_SIZE = [14.0, 14.0, 14.0]

# Initialize cached Vina engine
v = Vina(sf_name="vina", cpu=2, verbosity=0)
v.set_receptor(RECEPTOR)
v.compute_vina_maps(center=CENTER, box_size=BOX_SIZE)

def score_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return 0.0
        
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
            return 0.0

        conf = mol_h.GetConformer()
        centroid = AllChem.ComputeCentroid(conf)
        pocket = Point3D(CENTER[0], CENTER[1], CENTER[2])
        for idx in range(mol_h.GetNumAtoms()):
            p = conf.GetAtomPosition(idx)
            conf.SetAtomPosition(idx, Point3D(
                p.x - centroid.x + pocket.x,
                p.y - centroid.y + pocket.y,
                p.z - centroid.z + pocket.z
            ))

        prepper = MoleculePreparation()
        mol_setups = prepper.prepare(mol_h)
        lig_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

        v.set_ligand_from_string(lig_str)
        v.dock(exhaustiveness=2, n_poses=1, max_evals=100000)
        aff = float(v.energies(n_poses=1)[0][0])

        # Reward mapping:
        # -8.0 kcal/mol or better -> 1.0
        # -4.0 kcal/mol or worse  -> 0.0
        reward = max(0.0, min(1.0, (-aff - 4.0) / 4.0))
        return round(float(reward), 3)
    except Exception:
        return 0.0

if __name__ == "__main__":
    # If called with command-line SMILES or test run
    test_smi = "CN1CCN(c2cc3ncc(-c4cnn(C5CCNCC5)c4-c4ccc(F)cc4F)nc3cc2F)CC1"
    res = score_smiles(test_smi)
    print(f"Test docking score for Lead 0035: {res}")
