"""Mide la composición del grupo de control de la cohorte congelada (análisis post hoc).

Pregunta: ¿cuántos de los 4.639 controles tienen sospecha de infección
documentada, y cuántos cumplen además el criterio Sepsis-3 completo
(sospecha + SOFA >= 2 en la ventana de asociación)?

Reutiliza las funciones de la etapa 3 tal cual, para que la medición use
exactamente el mismo criterio que produjo las etiquetas. No reentrena nada
ni reescribe ningún artefacto de la cohorte.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "stage3", ROOT / "multi_modality_code" / "3_build_sepsis3_labels.py")
s3 = importlib.util.module_from_spec(spec)
sys.modules["stage3"] = s3
spec.loader.exec_module(s3)

args = argparse.Namespace(
    extracted_dir=str(ROOT / "data/01_extracted"),
    onset_dir=str(ROOT / "data/02_onset"),
    mapping_file=str(ROOT / "multi_modality_code/reference_files/measurement_mappings.json"),
    lead_time=6.0, sofa_threshold=2, min_los_hours=24.0, min_age=18.0,
    abx_before_culture_h=24.0, culture_before_abx_h=72.0, chunk_size=2_000_000,
)
search_dirs = [args.onset_dir, args.extracted_dir]

# --- controles de la cohorte congelada -------------------------------------
labels = pd.read_csv(ROOT / "data/03_labels/sepsis3_onset.csv", sep="|")
controls = set(labels.loc[labels.label == 0, "stay_id"].astype(int))
positives = set(labels.loc[labels.label == 1, "stay_id"].astype(int))
print(f"controles={len(controls)}  positivos={len(positives)}")

# --- elegibles (mismos filtros que build_cohort) ---------------------------
demog = s3.read_pipe_csv(s3.resolve_input("demog_processed.csv", search_dirs))
demog["stay_id"] = s3.to_int_stay(demog["stay_id"])
demog = demog.dropna(subset=["stay_id"]).copy()
demog["los_hours"] = pd.to_numeric(demog["los"], errors="coerce") * 24.0
elig = demog[(pd.to_numeric(demog["age"], errors="coerce") >= args.min_age)
             & (pd.to_numeric(demog["adm_order"], errors="coerce") == 1)
             & (demog["los_hours"] >= args.min_los_hours)].copy()
print(f"elegibles={len(elig)}  pool de candidatos (elegibles - positivos)="
      f"{len(set(elig.stay_id.astype(int)) - positives)}")

# --- sospecha de infección -------------------------------------------------
abx = s3.read_pipe_csv(s3.resolve_input("abx_processed.csv", search_dirs))
bact = s3.read_pipe_csv(s3.resolve_input("bacterio_processed.csv", search_dirs))
susp = s3.find_suspected_infection(abx, bact, args.abx_before_culture_h,
                                   args.culture_before_abx_h)
keys = elig[["subject_id", "stay_id", "hadm_id", "intime", "outtime",
             "age", "los_hours", "adm_order"]]
susp = susp.merge(keys, on="stay_id", how="inner", suffixes=("", "_d"))
susp_ids = set(susp.stay_id.astype(int))
print(f"elegibles con sospecha={len(susp_ids)}")

pool_ids = set(elig.stay_id.astype(int)) - positives
print(f"  de los cuales en el pool de candidatos={len(susp_ids & pool_ids)}")

susp_ctrl = susp[susp.stay_id.astype(int).isin(controls)].copy()
n_ctrl_susp = len(set(susp_ctrl.stay_id.astype(int)))
print(f"CONTROLES con sospecha de infección={n_ctrl_susp}"
      f"  ({100*n_ctrl_susp/len(controls):.1f} %)")

# --- SOFA >= 2 en la ventana de asociación, solo para esos controles -------
bounds = susp_ctrl[["stay_id", "suspected_infection_time", "intime", "outtime"]].copy()
bounds["lo_time"] = bounds.suspected_infection_time - s3.ASSOCIATION_BEFORE - s3.SOFA_WORST_WINDOW
bounds["hi_time"] = np.minimum(bounds.suspected_infection_time + s3.ASSOCIATION_AFTER,
                               bounds.outtime)
bounds = bounds[["stay_id", "lo_time", "hi_time"]]
print(f"cargando eventos de componentes SOFA para {len(bounds)} estancias...")
series = s3.load_component_events(search_dirs, args.mapping_file, bounds, args.chunk_size)

rows, n_meet = [], 0
for row in susp_ctrl.itertuples(index=False):
    sid = int(row.stay_id)
    susp_t, intime, outtime = float(row.suspected_infection_time), float(row.intime), float(row.outtime)
    g0 = max(susp_t - s3.ASSOCIATION_BEFORE, intime)
    g1 = min(susp_t + s3.ASSOCIATION_AFTER, outtime)
    meets, sofa_t, total = False, np.nan, np.nan
    if g1 >= g0:
        grid = np.arange(g0, g1 + 1.0, s3.HOUR)
        res = s3.compute_sofa_onset(series.get(sid, s3.StaySeries()), grid, args.sofa_threshold)
        if res is not None:
            sofa_t, _sub, total = res
            meets = True
            n_meet += 1
    rows.append({"stay_id": sid, "suspected_infection_time": susp_t,
                 "meets_sepsis3": meets, "sofa_time": sofa_t, "sofa_total": total})

out = pd.DataFrame(rows)
n = len(controls)
print()
print("=" * 62)
print(f"Controles totales                          {n}")
print(f"  con sospecha de infección                {n_ctrl_susp:5d}  ({100*n_ctrl_susp/n:5.1f} %)")
print(f"  que cumplen Sepsis-3 completo            {n_meet:5d}  ({100*n_meet/n:5.1f} %)")
print("=" * 62)

dest = ROOT / "data/05_results/control_arm_composition.csv"
out.to_csv(dest, index=False, float_format="%.17g")
summary = pd.DataFrame([{
    "n_controls": n, "n_controls_suspected_infection": n_ctrl_susp,
    "pct_controls_suspected_infection": round(100 * n_ctrl_susp / n, 2),
    "n_controls_meeting_sepsis3": n_meet,
    "pct_controls_meeting_sepsis3": round(100 * n_meet / n, 2),
    "n_eligible": len(elig), "n_positives": len(positives),
    "n_candidate_pool": len(pool_ids),
    "n_candidate_pool_suspected": len(susp_ids & pool_ids),
}])
summary.to_csv(ROOT / "data/05_results/control_arm_composition_summary.csv", index=False)
print(f"escrito {dest.name} ({len(out)} filas) y control_arm_composition_summary.csv")
