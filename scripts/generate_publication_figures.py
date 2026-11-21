#!/usr/bin/env python3
"""
Generates 6 publication-grade (300 DPI) figures for the KRAS G12D Switch-II
generative RL & exhaustive validation pipeline directly from output CSVs.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, rdFingerprintGenerator
from sklearn.decomposition import PCA

RDLogger.DisableLog("rdApp.*")

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#0f172a",
})

PALETTE = {
    "navy": "#1e3a8a",
    "teal": "#0d9488",
    "amber": "#d97706",
    "rose": "#e11d48",
    "slate": "#64748b",
    "bg": "#f8fafc",
}


def plot_pipeline_architecture():
    """Figure 1: Visual System & Multi-Physics Validation Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(13, 4.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    stages = [
        ("1. Generative RL\n(REINVENT 4)", "Prior: reinvent.prior\nDAP + Murcko Filter\n1,600 SMILES (50 steps)", "#dbeafe", "#1d4ed8"),
        ("2. MedChem & Novelty\nFilter Gate", "PAINS / Brenk / NIH\nCustom SMARTS Alerts\nECFP4 Tanimoto < 0.45", "#ccfbf1", "#0f766e"),
        ("3. 3D Physics &\nSwitch-II Docking", "pH 7.4 Protonation\nAutoDock Vina (exh=16)\nAsp12 Salt-Bridge Check", "#fef3c7", "#b45309"),
        ("4. Strain & WT\nCounter-Screen", "MMFF94s Bound Strain\n(< 6.0 kcal/mol)\nWT KRAS Gly12 ddG", "#ffe4e6", "#be123c"),
        ("5. Tier-1 Lead\nPrioritization", "Composite MPO Score\nBRICS Retrosynthesis\n3D PDBQT Poses", "#ede9fe", "#6d28d9"),
    ]

    xs = [2, 21.5, 41, 60.5, 80]
    w, h = 17.5, 22
    y = 7

    for i, ((title, body, fill, edge), x) in enumerate(zip(stages, xs)):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.8",
            facecolor=fill, edgecolor=edge, linewidth=1.8
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 4.5, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=edge)
        ax.plot([x + 1.5, x + w - 1.5], [y + h - 8.5, y + h - 8.5], color=edge, lw=0.8, alpha=0.5)
        ax.text(x + w / 2, y + 6.5, body, ha="center", va="center",
                fontsize=8.0, linespacing=1.45, color="#1e293b")
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.6, y + h / 2), xytext=(x + w + 0.6, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.6", color="#334155", lw=1.6))

    ax.text(50, 33.2, "KRAS G12D Switch-II Generative Design & Multi-Physics Validation Pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pipeline_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rl_trajectory(rl_csv: Path):
    """Figure 2: REINVENT 4 Learning Trajectory Across 50 RL Steps."""
    if not rl_csv.exists():
        return
    df = pd.read_csv(rl_csv)
    score_col = next((c for c in df.columns if c.lower() in ("score", "total_score")), None)
    if score_col is None:
        return
    df["Step"] = (np.arange(len(df)) // 32) + 1
    grp = df.groupby("Step")[score_col].agg(["mean", "max", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    ax.fill_between(grp["Step"], grp["mean"] - grp["std"], np.minimum(1.0, grp["mean"] + grp["std"]),
                    color=PALETTE["teal"], alpha=0.18, label="Batch ±1 SD")
    ax.plot(grp["Step"], grp["mean"], color=PALETTE["teal"], lw=2.3, label="Batch Mean Reward")
    ax.plot(grp["Step"], grp["max"], color=PALETTE["navy"], lw=1.6, ls="--", label="Step Peak Candidate")

    ax.set_title("REINVENT 4 Policy Optimization Trajectory (KRAS G12D Switch-II)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Reinforcement Learning Step (32 molecules / batch)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Multi-Parameter Reward Score", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rl_learning_trajectory.png", dpi=300)
    plt.close()


def plot_funnel_attrition(audit_df: pd.DataFrame):
    """Figure 3: Waterfall / Horizontal Attrition Funnel."""
    total_raw = 1600
    n_valid = len(audit_df)
    n_clean = int(audit_df["Alerts_Clean"].sum()) if "Alerts_Clean" in audit_df else n_valid
    n_novel = int((audit_df["Alerts_Clean"] & audit_df["Novel_Scaffold_Pass"]).sum()) if "Novel_Scaffold_Pass" in audit_df else n_clean
    n_docked = int(audit_df["G12D_Vina_kcal"].notna().sum()) if "G12D_Vina_kcal" in audit_df else 0
    n_tier1 = int(((audit_df["G12D_Vina_kcal"] <= -8.5) & audit_df["Alerts_Clean"]).sum()) if n_docked > 0 else 0

    labels = [
        "Raw RL Generated SMILES",
        "Unique Sanitized Structures",
        "PAINS / Brenk / NIH Clean",
        "Novel Scaffolds (Tanimoto < 0.45)",
        "High-Exhaustiveness 3D Docked",
        "Tier-1 Leads (Vina ≤ -8.5 kcal/mol)",
    ]
    counts = [total_raw, n_valid, n_clean, n_novel, n_docked, n_tier1]
    colors = ["#1e3a8a", "#2563eb", "#0d9488", "#059669", "#d97706", "#e11d48"]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=300)
    y_pos = np.arange(len(labels))[::-1]
    bars = ax.barh(y_pos, counts, color=colors, height=0.62, edgecolor="#1e293b", linewidth=0.8)

    for bar, c in zip(bars, counts):
        pct = (c / total_raw) * 100
        ax.text(bar.get_width() + 25, bar.get_y() + bar.get_height() / 2,
                f"{c:,} ({pct:.1f}%)", va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.22)
    ax.set_xlabel("Compound Count", fontsize=10, fontweight="bold")
    ax.set_title("Multi-Stage In Silico Validation & Filtering Funnel", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "validation_funnel.png", dpi=300)
    plt.close()


def plot_multi_physics_dashboard(audit_df: pd.DataFrame):
    """Figure 4: 2x2 Physics & Medicinal Chemistry Dashboard (Affinity, Asp12 Distance, Strain, QED/SA)."""
    docked = audit_df[audit_df["G12D_Vina_kcal"].notna()].copy() if "G12D_Vina_kcal" in audit_df else pd.DataFrame()
    if docked.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=300)

    # Panel A: Affinity vs QED colored by Asp12 Distance
    ax = axes[0, 0]
    sc = ax.scatter(
        docked["G12D_Vina_kcal"], docked["QED"],
        c=docked["Asp12_Min_Dist_A"].fillna(7.0), cmap="viridis_r",
        s=48, alpha=0.88, edgecolors="k", linewidths=0.4, vmin=2.5, vmax=6.5
    )
    ax.axvline(-8.5, color=PALETTE["rose"], ls="--", lw=1.2, label="Tier-1 Gate (-8.5 kcal/mol)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Min Distance to Asp12 (Å)", fontsize=8.5)
    ax.set_xlabel("KRAS G12D Vina ΔG (kcal/mol)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("QED Drug-Likeness", fontsize=9.5, fontweight="bold")
    ax.set_title("A. Potency vs. Drug-Likeness & Asp12 Contact", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.45)

    # Panel B: Bound Conformational Strain vs Vina Affinity
    ax = axes[0, 1]
    strain_valid = docked[docked["Bound_Strain_kcal"].notna()]
    ax.scatter(strain_valid["Bound_Strain_kcal"], strain_valid["G12D_Vina_kcal"],
               color=PALETTE["teal"], s=45, alpha=0.85, edgecolors="k", linewidths=0.4)
    ax.axvline(6.0, color=PALETTE["rose"], ls="--", lw=1.2, label="Max Strain Limit (6.0 kcal/mol)")
    ax.axhline(-8.5, color=PALETTE["navy"], ls=":", lw=1.2)
    ax.set_xlabel("Bound MMFF94s Strain Energy ΔE (kcal/mol)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("KRAS G12D Vina ΔG (kcal/mol)", fontsize=9.5, fontweight="bold")
    ax.set_title("B. Pose Realism: Affinity vs. Internal Strain", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.45)

    # Panel C: Asp12 Salt-Bridge Distance Distribution
    ax = axes[1, 0]
    dists = docked["Asp12_Min_Dist_A"].dropna()
    ax.hist(dists, bins=18, color=PALETTE["navy"], alpha=0.8, edgecolor="white")
    ax.axvline(4.2, color=PALETTE["rose"], ls="--", lw=1.5, label="Salt-Bridge / H-Bond Cutoff (4.2 Å)")
    ax.set_xlabel("Minimum Polar Distance to Asp12 OD1/OD2 (Å)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Candidate Count", fontsize=9.5, fontweight="bold")
    ax.set_title("C. Oncogenic Asp12 Anchor Engagement", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.45)

    # Panel D: Novelty (Max Tanimoto) vs Synthetic Accessibility
    ax = axes[1, 1]
    ax.scatter(docked["Max_Ref_Tanimoto"], docked["SAScore"],
               c=docked["Composite_MPO_Score"], cmap="plasma", s=48, alpha=0.88, edgecolors="k", linewidths=0.4)
    ax.axvline(0.45, color=PALETTE["rose"], ls="--", lw=1.2, label="Novelty Cutoff (Tanimoto < 0.45)")
    ax.axhline(4.0, color=PALETTE["slate"], ls=":", lw=1.2, label="Easy Synthesis (SAScore < 4.0)")
    ax.set_xlabel("Max ECFP4 Tanimoto vs. Known KRAS Drugs", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Synthetic Accessibility (SAScore)", fontsize=9.5, fontweight="bold")
    ax.set_title("D. IP Novelty vs. Synthetic Feasibility", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.45)

    plt.suptitle("Multi-Physics & Medicinal Chemistry Audit Dashboard (KRAS G12D)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "multi_physics_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_chemical_space_pca(audit_df: pd.DataFrame):
    """Figure 5: ECFP4 Chemical Space PCA Showing Novel RL Scaffolds vs. Reference Inhibitors."""
    refs = {
        "MRTX1133": "C[C@H]1CN(c2nc(OC[C@@]34CCCN3C[C@H](F)C4)nc3c2CCN(c2cccc4cccc(Cl)c24)C3)CCN1",
        "HRS-4642_analog": "Fc1ccc2c(c1)c(N1CCNCC1)nc(OC[C@]13CCCN1CC(F)C3)n2",
        "TH-Z835": "CC(C)c1cccc(C(C)C)c1-n1cnc2c(N3CCN(C(=O)C=C)CC3)nc(OC[C@H]3CCCN3C)nc21",
        "BI-2852": "CN(C)Cc1ccc(-c2nc3cc(O)ccc3[nH]2)cc1",
    }
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    sample_df = audit_df.head(400).copy()
    fps, is_top = [], []
    for i, row in sample_df.iterrows():
        m = Chem.MolFromSmiles(row["Canonical_SMILES"])
        if m:
            fps.append(np.array(gen.GetFingerprint(m)))
            is_top.append(i < 25)

    ref_fps, ref_names = [], []
    for name, smi in refs.items():
        m = Chem.MolFromSmiles(smi)
        if m:
            ref_fps.append(np.array(gen.GetFingerprint(m)))
            ref_names.append(name)

    if len(fps) < 10:
        return

    all_fps = np.vstack([fps, ref_fps])
    coords = PCA(n_components=2, random_state=42).fit_transform(all_fps)
    gen_coords = coords[:len(fps)]
    ref_coords = coords[len(fps):]
    is_top = np.array(is_top)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=300)
    ax.scatter(gen_coords[~is_top, 0], gen_coords[~is_top, 1], c="#94a3b8", alpha=0.45, s=28, label="Generated RL Library")
    ax.scatter(gen_coords[is_top, 0], gen_coords[is_top, 1], c=PALETTE["teal"], alpha=0.9, s=65, edgecolors="k", linewidths=0.5, label="Top 25 MPO Candidates")
    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], c=PALETTE["rose"], marker="*", s=220, edgecolors="k", linewidths=0.8, label="Known KRAS Inhibitors")

    for (x, y), name in zip(ref_coords, ref_names):
        ax.annotate(name, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8.5, fontweight="bold", color=PALETTE["rose"])

    ax.set_title("ECFP4 Chemical Space Map: Novel RL Scaffolds vs. Clinical/Preclinical Reference Ligands", fontsize=11, fontweight="bold")
    ax.set_xlabel("Principal Component 1 (ECFP4 Structural Variance)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Principal Component 2 (ECFP4 Structural Variance)", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "chemical_space_pca.png", dpi=300)
    plt.close()


def plot_top_leads_grid(audit_df: pd.DataFrame):
    """Figure 6: 2D Chemical Structure Grid of Top 8 MPO-Ranked Leads with Full Metrics."""
    top = audit_df.head(8)
    mols, legends = [], []
    for _, row in top.iterrows():
        m = Chem.MolFromSmiles(row["Canonical_SMILES"])
        if m:
            mols.append(m)
            aff = f"{row['G12D_Vina_kcal']:.2f}" if pd.notna(row.get("G12D_Vina_kcal")) else "N/A"
            asp = f"{row['Asp12_Min_Dist_A']:.2f}Å" if pd.notna(row.get("Asp12_Min_Dist_A")) else "N/A"
            legends.append(
                f"#{int(row['Benchmark_Rank'])} | MPO: {row['Composite_MPO_Score']:.3f}\n"
                f"ΔG: {aff} kcal/mol | Asp12: {asp}\n"
                f"QED: {row['QED']:.2f} | SA: {row['SAScore']:.2f} | Sim: {row['Max_Ref_Tanimoto']:.2f}"
            )
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(340, 290), legends=legends)
        img.save(str(FIG_DIR / "top_leads_grid.png"))


def main():
    rl_csv = Path("results/kras_g12d_balanced_1.csv")
    audit_csv = Path("exhaustive_benchmark_out/full_1600_audited_library.csv")

    plot_pipeline_architecture()
    plot_rl_trajectory(rl_csv)

    if audit_csv.exists():
        audit_df = pd.read_csv(audit_csv)
        plot_funnel_attrition(audit_df)
        plot_multi_physics_dashboard(audit_df)
        plot_chemical_space_pca(audit_df)
        plot_top_leads_grid(audit_df)
        print("[OK] Generated all 6 publication figures in figures/")
    else:
        print("[INFO] Generated architecture & RL trajectory plots. Run exhaustive_kras_benchmark.py first to populate physics plots.")


if __name__ == "__main__":
    main()
