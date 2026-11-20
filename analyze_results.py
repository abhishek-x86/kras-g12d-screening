import os
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

os.makedirs("figures", exist_ok=True)

vs_df = pd.read_csv("kras_screening_output/kras_final_ranked_leads.csv")
rl_df = pd.read_csv("results/kras_g12d_rl_1.csv")

rl_df["Est_Affinity_kcal_mol"] = -(rl_df["Vina_Docking_KRAS_G12D (raw)"] * 6.0 + 3.0)
rl_df["QED"] = rl_df["Drug_Likeness_QED (raw)"]

# 1. Pareto Space: Binding Affinity vs. QED
plt.figure(figsize=(8, 5.5), dpi=300)
vs_favorable = vs_df[vs_df["Binding_Affinity_kcal_mol"] < 0]

plt.scatter(
    vs_favorable["QED"], vs_favorable["Binding_Affinity_kcal_mol"],
    c="#2b5c8f", alpha=0.8, s=65, label="Targeted Library (Vina Exhaustiveness=4)"
)
plt.scatter(
    rl_df["QED"], rl_df["Est_Affinity_kcal_mol"],
    c="#d95f02", alpha=0.75, s=65, marker="^", label="REINVENT 4 RL Candidates"
)
plt.axhline(-7.29, color="#1b9e77", linestyle="--", linewidth=1.5, label="MRTX1133 Reference (-7.29 kcal/mol)")

plt.xlabel("Quantitative Estimate of Drug-Likeness (QED)", fontsize=11)
plt.ylabel("Predicted Binding Free Energy ΔG (kcal/mol)", fontsize=11)
plt.title("KRAS G12D Switch-II Pocket: Binding Affinity vs. Drug-Likeness", fontsize=12, fontweight="bold")
plt.gca().invert_yaxis()
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig("figures/pareto_affinity_vs_qed.png")
plt.close()

# 2. RL Optimization Trajectory Across Steps
if "step" in rl_df.columns and rl_df["step"].nunique() > 1:
    step_stats = rl_df.groupby("step").agg(
        mean_score=("Score", "mean"),
        max_score=("Score", "max"),
        mean_aff=("Est_Affinity_kcal_mol", "mean"),
        best_aff=("Est_Affinity_kcal_mol", "min")
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    ax1.plot(step_stats["step"], step_stats["mean_score"], marker="o", color="#2b5c8f", label="Batch Mean Reward")
    ax1.plot(step_stats["step"], step_stats["max_score"], marker="s", linestyle="--", color="#1b9e77", label="Step Peak Reward")
    ax1.set_xlabel("REINVENT 4 RL Step", fontsize=10)
    ax1.set_ylabel("Multi-Parameter Score [0–1]", fontsize=10)
    ax1.set_title("Agent Reward Progression", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(step_stats["step"], step_stats["mean_aff"], marker="o", color="#d95f02", label="Batch Mean ΔG")
    ax2.plot(step_stats["step"], step_stats["best_aff"], marker="s", linestyle="--", color="#7570b3", label="Step Best ΔG")
    ax2.axhline(-7.29, color="#1b9e77", linestyle=":", linewidth=1.2, label="MRTX1133 (-7.29)")
    ax2.set_xlabel("REINVENT 4 RL Step", fontsize=10)
    ax2.set_ylabel("Predicted Binding ΔG (kcal/mol)", fontsize=10)
    ax2.set_title("Switch-II Binding Affinity Trajectory", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/rl_learning_trajectory.png")
    plt.close()

# 3. 2D Structure Grid of Top 6 Combined Hits
top_vs = vs_favorable[["Compound_ID", "SMILES", "Binding_Affinity_kcal_mol", "QED"]].head(3).copy()
top_vs.columns = ["ID", "SMILES", "Affinity", "QED"]

rl_top = rl_df.drop_duplicates(subset=["SMILES"]).sort_values(by="Score", ascending=False).head(3).copy()
rl_top["ID"] = [f"RL_STEP{int(s)}_{i+1:02d}" for i, s in enumerate(rl_top["step"])]
rl_top = rl_top[["ID", "SMILES", "Est_Affinity_kcal_mol", "QED"]]
rl_top.columns = ["ID", "SMILES", "Affinity", "QED"]

combined_top = pd.concat([top_vs, rl_top], ignore_index=True)
mols, legends = [], []
for _, row in combined_top.iterrows():
    m = Chem.MolFromSmiles(row["SMILES"])
    if m:
        mols.append(m)
        legends.append(f"{row['ID']}\nΔG: {row['Affinity']:.2f} kcal/mol | QED: {row['QED']:.2f}")

img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(320, 260), legends=legends)
img.save("figures/top_leads_grid.png")
print(f"[✓] Processed {len(rl_df)} RL molecules across {rl_df['step'].nunique()} steps and saved figures!")
