import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, Draw

val_df = pd.read_csv("results/validated_top_hits.csv")
targets = val_df[val_df["Candidate"].isin(["RL_Top_01", "RL_Top_02", "RL_Top_03", "MRTX1133_Ref"])].copy()

print("=== Retrosynthetic Building-Block Breakdown (BRICS) ===\n")

records = []
all_frag_mols = []
all_frag_legends = []

for row in targets.itertuples():
    mol = Chem.MolFromSmiles(row.SMILES)
    # Break molecule at standard medicinal-chemistry synthesis bonds
    fragments = sorted(BRICS.BRICSDecompose(mol))
    num_steps = max(1, len(fragments) - 1)

    records.append({
        "Candidate": row.Candidate,
        "Strict_DeltaG": row.Strict_DeltaG_kcal_mol,
        "Building_Blocks_Count": len(fragments),
        "Est_Assembly_Steps": num_steps,
        "Building_Blocks_SMILES": " + ".join(fragments)
    })

    print(f"{row.Candidate:<15} | ΔG: {row.Strict_DeltaG_kcal_mol:>6} | Bricks Needed: {len(fragments)} | Est. Synthesis Steps: {num_steps}")
    for i, frag in enumerate(fragments, 1):
        print(f"   Brick {i}: {frag}")
        fm = Chem.MolFromSmiles(frag)
        if fm and len(all_frag_mols) < 12:
            all_frag_mols.append(fm)
            all_frag_legends.append(f"{row.Candidate} - Brick {i}")
    print()

pd.DataFrame(records).to_csv("results/retrosynthesis_routes.csv", index=False)

if all_frag_mols:
    img = Draw.MolsToGridImage(all_frag_mols, molsPerRow=4, subImgSize=(280, 220), legends=all_frag_legends)
    img.save("figures/building_blocks_grid.png")
    print("Saved results/retrosynthesis_routes.csv and figures/building_blocks_grid.png")
