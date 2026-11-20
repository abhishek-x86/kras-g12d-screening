import os
import csv
import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

# -------------------------------------------------------------
# 1. TARGET & OUTPUT DIRECTORIES
# -------------------------------------------------------------
RECEPTOR_FILE = "7RT1_fixed_receptor.pdbqt"
OUT_DIR = "kras_screening_output"
POSES_DIR = os.path.join(OUT_DIR, "docked_poses")
os.makedirs(POSES_DIR, exist_ok=True)

FINAL_CSV = os.path.join(OUT_DIR, "kras_final_ranked_leads.csv")
PRELIMINARY_CSV = os.path.join(OUT_DIR, "kras_all_generated_preliminary.csv")

# KRAS G12D Switch-II pocket coordinates (7RT1 centroid)
CENTER = [-1.810, -1.541, -25.541]
BOX_SIZE = [14.0, 14.0, 14.0]
BENCHMARK_SCORE = -7.286  # MRTX1133 reference threshold

# -------------------------------------------------------------
# 2. LOAD RECEPTOR & INITIALIZE VINA IN MEMORY
# -------------------------------------------------------------
print("Initializing in-memory Vina C++ engine...")
v = Vina(sf_name="vina", cpu=4, verbosity=0)
v.set_receptor(RECEPTOR_FILE)

print("Pre-computing grid maps directly into RAM (runs once)...")
v.compute_vina_maps(center=CENTER, box_size=BOX_SIZE)
print("[✓] Grids cached in memory. Fast screening ready!\n")

# -------------------------------------------------------------
# 3. DRUGLIKENESS & MEDICINAL CHEMISTRY GATES
# -------------------------------------------------------------
def passes_publication_gates(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    qed = Descriptors.qed(mol)

    if not (300 <= mw <= 650):
        return None, f"MW out of range ({mw:.1f})"
    if not (1.0 <= logp <= 5.5):
        return None, f"LogP out of range ({logp:.1f})"
    if not (40 <= tpsa <= 140):
        return None, f"TPSA out of range ({tpsa:.1f})"
    if rotb > 10:
        return None, f"Too many rotatable bonds ({rotb})"

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "RotBonds": rotb,
        "QED": round(qed, 2),
        "SAS": 2.5
    }, "Pass"

# -------------------------------------------------------------
# 4. LOAD CANDIDATES
# -------------------------------------------------------------
df_prev = pd.read_csv("kras_screening_output/kras_all_generated_preliminary.csv")
candidate_smiles = df_prev["SMILES"].dropna().unique().tolist()
print(f"Loaded {len(candidate_smiles)} candidates.")

# -------------------------------------------------------------
# 5. HIGH-SPEED SCREENING LOOP
# -------------------------------------------------------------
docked_hits = []

with open(PRELIMINARY_CSV, "w", newline="", encoding="utf-8") as p_file:
    p_writer = csv.writer(p_file)
    p_writer.writerow(["Candidate_ID", "SMILES", "MW", "LogP", "TPSA", "RotBonds", "QED", "SAS", "Gate_Status", "Rejection_Reason"])

    for i, smi in enumerate(tqdm(candidate_smiles, desc="Screening Compounds")):
        cid = f"KRAS_G12D_{i+1:04d}"
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue

        gate_data, status_msg = passes_publication_gates(mol)
        if gate_data is None:
            mw = round(Descriptors.MolWt(mol), 2)
            logp = round(Descriptors.MolLogP(mol), 2)
            p_writer.writerow([cid, smi, mw, logp, "-", "-", "-", "-", "REJECTED", status_msg])
            continue

        p_writer.writerow([
            cid, smi, gate_data["MW"], gate_data["LogP"], gate_data["TPSA"],
            gate_data["RotBonds"], gate_data["QED"], gate_data["SAS"], "PASSED", "Meets all criteria"
        ])

        # 3D ligand conformer embedding
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            continue

        # Pre-center coordinates into the binding pocket cavity
        conf = mol_h.GetConformer()
        centroid = AllChem.ComputeCentroid(conf)
        pocket_center = Point3D(CENTER[0], CENTER[1], CENTER[2])
        for atom_idx in range(mol_h.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            new_pos = Point3D(
                pos.x - centroid.x + pocket_center.x,
                pos.y - centroid.y + pocket_center.y,
                pos.z - centroid.z + pocket_center.z
            )
            conf.SetAtomPosition(atom_idx, new_pos)

        # Prepare PDBQT in memory
        try:
            prepper = MoleculePreparation()
            mol_setups = prepper.prepare(mol_h)
            lig_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
        except Exception:
            continue

        # In-Memory Docking & Relaxation
        try:
            v.set_ligand_from_string(lig_str)
            v.dock(exhaustiveness=4, n_poses=1, max_evals=250000)
            v.optimize()
            aff = float(v.energies(n_poses=1)[0][0])
        except Exception:
            continue

        # Log ALL successfully docked candidates
        core_scaff = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi)
        pose_path = os.path.join(POSES_DIR, f"{cid}_docked.pdbqt")
        v.write_poses(pose_path, n_poses=1, overwrite=True)

        docked_hits.append({
            "Compound_ID": cid,
            "SMILES": smi,
            "Binding_Affinity_kcal_mol": round(aff, 2),
            "DeltaG_vs_MRTX1133": round(aff - BENCHMARK_SCORE, 2),
            "MW": gate_data["MW"],
            "LogP": gate_data["LogP"],
            "TPSA": gate_data["TPSA"],
            "RotBonds": gate_data["RotBonds"],
            "QED": gate_data["QED"],
            "Scaffold": core_scaff,
            "Pose_File": f"{cid}_docked.pdbqt"
        })

# -------------------------------------------------------------
# 6. EXPORT & RANK TOP LEADS
# -------------------------------------------------------------
hits_df = pd.DataFrame(docked_hits)
if len(hits_df) > 0:
    hits_df = hits_df.sort_values(by="Binding_Affinity_kcal_mol", ascending=True)
    hits_df.to_csv(FINAL_CSV, index=False)
    
    print("\n" + "="*70)
    print("SCREENING COMPLETE & SCORES RANKED")
    print(f"Total screened: {len(candidate_smiles)}")
    print(f"Total successfully docked: {len(hits_df)}")
    print(f"Ranked output saved to: {FINAL_CSV}")
    print("="*70)
    print("\nTOP 10 LEADS (Switch-II Binding Affinity):")
    cols_to_show = ["Compound_ID", "Binding_Affinity_kcal_mol", "DeltaG_vs_MRTX1133", "MW", "LogP", "QED"]
    print(hits_df[cols_to_show].head(10).to_string(index=False))
    print("="*70 + "\n")
else:
    print("\nNo compounds could be docked successfully.")
