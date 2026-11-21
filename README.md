# Structure-Based Generative RL & Multi-Physics Validation of Non-Covalent KRAS G12D Inhibitors

[![Target](https://img.shields.io/badge/Target-KRAS_G12D_(PDB:_7RT1)-1e3a8a)]()
[![Generator](https://img.shields.io/badge/Generator-REINVENT_4_RL-0d9488)]()
[![Physics](https://img.shields.io/badge/Validation-Vina_+_MMFF94s_+_WT_ddG-be123c)]()

End-to-end generative drug discovery and multi-physics auditing pipeline targeting the **KRAS G12D Switch-II allosteric pocket** and mutant **Asp12** carboxylate anchor.

---

## 1. Pipeline Architecture

![Pipeline Architecture](figures/pipeline_architecture.png)

---

## 2. Multi-Stage Validation Funnel & RL Optimization

| Multi-Stage Attrition Funnel | REINVENT 4 Policy Learning Trajectory |
| :---: | :---: |
| ![Validation Funnel](figures/validation_funnel.png) | ![RL Trajectory](figures/rl_learning_trajectory.png) |

---

## 3. Multi-Physics & Medicinal Chemistry Audit

![Multi-Physics Dashboard](figures/multi_physics_dashboard.png)

---

## 4. Chemical Novelty Space & Top Validated Leads

| ECFP4 Chemical Space Map vs. Clinical Inhibitors | Top 8 MPO-Ranked Novel Leads |
| :---: | :---: |
| ![Chemical Space PCA](figures/chemical_space_pca.png) | ![Top Leads Grid](figures/top_leads_grid.png) |

---

## 5. Quickstart

```bash
# 1. Run exhaustive chemoinformatics, pH 7.4 protonation, strain & WT selectivity audit
python scripts/exhaustive_kras_benchmark.py \
  --input results/kras_g12d_balanced_1.csv \
  --receptor receptors/7RT1_fixed_receptor.pdbqt \
  --wt-receptor receptors/7RT1_wt_gly12.pdbqt \
  --top-dock 150 \
  --exhaustiveness 16 \
  --workers 4

# 2. Regenerate all 300-DPI figures in figures/
python scripts/generate_publication_figures.py
