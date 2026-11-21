import os
import importlib.util
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, RDConfig

sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
spec = importlib.util.spec_from_file_location("sascorer", sa_path)
sascorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sascorer)

params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
catalog = FilterCatalog.FilterCatalog(params)

val_df = pd.read_csv("results/validated_top_hits.csv")

rows = []
print("=== Route-Agnostic Preclinical Audit (IV / Subcutaneous / Oral) ===\n")

for row in val_df.itertuples():
    mol = Chem.MolFromSmiles(row.SMILES)
    if not mol:
        continue

    sa = round(sascorer.calculateScore(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(Descriptors.TPSA(mol), 1)
    matches = [entry.GetDescription() for entry in catalog.GetMatches(mol)]
    alerts = "; ".join(matches) if matches else "CLEAN (0 alerts)"

    # Route-agnostic viability: synthesizable, alert-free, and soluble enough for IV/SC formulation
    iv_soluble = (-1.0 <= logp <= 5.5)
    makable = (sa <= 4.0)
    safe_motif = (len(matches) == 0)
    viable = (iv_soluble and makable and safe_motif)

    suggested_route = "Oral / IV / SC" if (row.MW <= 500 and tpsa <= 140 and iv_soluble) else "Parenteral (IV / Subcutaneous)"

    rows.append({
        "Candidate": row.Candidate,
        "Strict_DeltaG": row.Strict_DeltaG_kcal_mol,
        "MW": row.MW,
        "LogP": logp,
        "SA_Score": sa,
        "Structural_Alerts": alerts,
        "Suggested_Route": suggested_route,
        "Preclinical_Viable": "PASS" if viable else "FLAGGED"
    })

    print(f"{row.Candidate:<15} | ΔG: {row.Strict_DeltaG_kcal_mol:>6} | SA: {sa} | LogP: {logp} | Route: {suggested_route}")
    print(f"   Status: {'PASS' if viable else 'FLAGGED'} | Alerts: {alerts}\n")

pd.DataFrame(rows).to_csv("results/medchem_audit_report.csv", index=False)
print("Saved results/medchem_audit_report.csv")
