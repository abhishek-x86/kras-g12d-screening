from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

RECEPTOR = "7RT1_fixed_receptor.pdbqt"
CENTER = [-1.810, -1.541, -25.541]
BOX_SIZE = [18.0, 18.0, 18.0]
OUT_DIR = Path("results/validated_poses")
OUT_DIR.mkdir(parents=True, exist_ok=True)

v = Vina(sf_name="vina", cpu=4, verbosity=0)
v.set_receptor(RECEPTOR)
v.compute_vina_maps(center=CENTER, box_size=BOX_SIZE)

def strict_dock(name, smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        return None

    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)

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
    mol_setups = prepper.prepare(mol_h)
    lig_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

    v.set_ligand_from_string(lig_str)
    v.dock(exhaustiveness=16, n_poses=1)
    v.optimize()
    aff = float(v.energies(n_poses=1)[0][0])

    pose_file = OUT_DIR / f"{name}_strict.pdbqt"
    v.write_poses(str(pose_file), n_poses=1, overwrite=True)

    return {
        "Candidate": name,
        "Strict_DeltaG_kcal_mol": round(aff, 2),
        "MW": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "QED": round(Descriptors.qed(mol), 2),
        "SMILES": smi
    }

rl_df = pd.read_csv("results/kras_g12d_rl_1.csv")
top_rl = rl_df.drop_duplicates(subset=["SMILES"]).sort_values(by="Score", ascending=False).head(5)

candidates = [
    ("MRTX1133_Ref", "CN1CCN(CC1)c2ccc3nc(c(cc3c2)F)c4c(cnn4C5CCNCC5)c6c(ccc(c6F)F)O"),
    ("KRAS_G12D_0035", "CN1CCN(c2cc3ncc(-c4cnn(C5CCNCC5)c4-c4ccc(F)cc4F)nc3cc2F)CC1"),
]
for i, row in enumerate(top_rl.itertuples(), 1):
    candidates.append((f"RL_Top_{i:02d}", row.SMILES))

print("Running strict validation (takes ~1-2 minutes)...")
records = []
for name, smi in candidates:
    res = strict_dock(name, smi)
    if res:
        records.append(res)
        print(f"  [✓] {name:<15} -> {res['Strict_DeltaG_kcal_mol']} kcal/mol (QED: {res['QED']})")

val_df = pd.DataFrame(records).sort_values(by="Strict_DeltaG_kcal_mol", ascending=True)
val_df.to_csv("results/validated_top_hits.csv", index=False)
print("\nDone! Saved results/validated_top_hits.csv")
