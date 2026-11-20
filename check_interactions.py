from pathlib import Path
import numpy as np
import pandas as pd

def parse_pdb_atoms(path, is_ligand=False):
    atoms = []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if is_ligand and line.startswith("ATOM"):
                    pass
                name = line[12:16].strip()
                resname = line[17:20].strip()
                resnum = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() or name[0]
                atoms.append((resname, resnum, name, elem, np.array([x, y, z])))
    return atoms

receptor_atoms = parse_pdb_atoms("7RT1_fixed_receptor.pdbqt")
val_df = pd.read_csv("results/validated_top_hits.csv")

print("=== Switch-II Pocket Residue Contacts (< 4.0 Å) ===\n")
for row in val_df.itertuples():
    pose_path = Path(f"results/validated_poses/{row.Candidate}_strict.pdbqt")
    if not pose_path.exists():
        continue
    lig_atoms = parse_pdb_atoms(pose_path, is_ligand=True)

    contacts = {}
    for r_res, r_num, r_name, r_elem, r_xyz in receptor_atoms:
        for _, _, l_name, l_elem, l_xyz in lig_atoms:
            dist = float(np.linalg.norm(r_xyz - l_xyz))
            if dist <= 4.0:
                key = f"{r_res}{r_num}"
                is_polar = (r_elem in {"N", "O", "S"} and l_elem in {"N", "O", "S", "F", "Cl"})
                if key not in contacts or dist < contacts[key][0]:
                    contacts[key] = (dist, "Polar/H-bond" if is_polar else "Hydrophobic")

    sorted_c = sorted(contacts.items(), key=lambda x: x[1][0])
    asp12 = contacts.get("ASP12", None)
    asp12_str = f"YES ({asp12[0]:.2f} Å, {asp12[1]})" if asp12 else "No (<4.0 Å)"
    top_res = ", ".join([f"{k} ({v[0]:.1f}Å)" for k, v in sorted_c[:6]])

    print(f"{row.Candidate:<15} | ΔG: {row.Strict_DeltaG_kcal_mol:>6} | Asp12 Contact: {asp12_str}")
    print(f"   Closest residues: {top_res}\n")
