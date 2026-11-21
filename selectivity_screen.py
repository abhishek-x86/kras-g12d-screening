import subprocess
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

CENTER = [-1.810, -1.541, -25.541]
BOX_SIZE = [18.0, 18.0, 18.0]
WT_PDB = Path("7RT1_wt_gly12.pdb")
WT_PDBQT = Path("7RT1_wt_gly12.pdbqt")

# 1. Construct in silico Wild-Type KRAS (revert Asp12 side chain -> Gly12)
wt_lines = []
with open("7RT1_fixed_receptor.pdbqt") as f:
    for line in f:
        if line.startswith("ATOM"):
            resname = line[17:20].strip()
            resnum = line[22:26].strip()
            atom_name = line[12:16].strip()
            # Skip Asp12 side-chain atoms to revert residue 12 to Glycine
            if resname == "ASP" and resnum == "12" and atom_name not in {"N", "CA", "C", "O", "H", "HA", "HN"}:
                continue
            if resname == "ASP" and resnum == "12":
                line = line[:17] + "GLY" + line[20:]
            # Strip PDBQT charge/type columns to write clean PDB coordinates
            wt_lines.append(line[:54] + "\n")

with open(WT_PDB, "w") as out:
    out.writelines(wt_lines)
    out.write("END\n")

subprocess.run(["obabel", str(WT_PDB), "-O", str(WT_PDBQT), "-xr", "-p", "7.4", "-h"], check=True)

# 2. Initialize Vina on Healthy Wild-Type KRAS
v_wt = Vina(sf_name="vina", cpu=4, verbosity=0)
v_wt.set_receptor(str(WT_PDBQT))
v_wt.compute_vina_maps(center=CENTER, box_size=BOX_SIZE)

def dock_wt(smi):
    mol = Chem.MolFromSmiles(smi)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol_h, params)
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
    lig_str = PDBQTWriterLegacy.write_string(prepper.prepare(mol_h)[0])[0]
    v_wt.set_ligand_from_string(lig_str)
    v_wt.dock(exhaustiveness=16, n_poses=1)
    v_wt.optimize()
    return round(float(v_wt.energies(n_poses=1)[0][0]), 2)

# 3. Compare Mutant G12D vs. Wild-Type Gly12 Affinity
val_df = pd.read_csv("results/validated_top_hits.csv")
targets = val_df[val_df["Candidate"].isin(["RL_Top_01", "RL_Top_02", "RL_Top_03", "RL_Top_05", "MRTX1133_Ref"])].copy()

print("=== Mutant KRAS (G12D) vs. Healthy Wild-Type KRAS (Gly12) Selectivity ===\n")
records = []
for row in targets.itertuples():
    wt_aff = dock_wt(row.SMILES)
    delta_sel = round(row.Strict_DeltaG_kcal_mol - wt_aff, 2)  # Negative = selective for cancer mutant
    verdict = "SELECTIVE (G12D > WT)" if delta_sel < -0.3 else "NON-SELECTIVE"
    records.append({
        "Candidate": row.Candidate,
        "G12D_DeltaG": row.Strict_DeltaG_kcal_mol,
        "WT_Gly12_DeltaG": wt_aff,
        "Selectivity_ddG": delta_sel,
        "Verdict": verdict
    })
    print(f"{row.Candidate:<15} | G12D: {row.Strict_DeltaG_kcal_mol:>6} | WT: {wt_aff:>6} | ΔΔG: {delta_sel:>+6} -> {verdict}")

pd.DataFrame(records).to_csv("results/selectivity_counter_screen.csv", index=False)
print("\nSaved results/selectivity_counter_screen.csv")
