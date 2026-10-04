"""Mide la exposición a antibióticos en la ventana de observación (análisis post hoc).

Pregunta: ¿en qué proporción de cada brazo hay un intervalo de antibiótico que
solapa la ventana de 24 h previa a t0, y dónde cae el antibiótico que define la
sospecha de infección de los positivos respecto a t0 y al onset?

Explica el mecanismo de la ablación sensitivity_with_antibiotics. La exposición
se lee de la variante _abx del dataset multimodal, que aplica ya la regla de
solapamiento de la etapa 4 (starttime <= prediction_time y endtime >= lower_bound);
los tiempos, de las etiquetas de la etapa 3. No reentrena nada ni reescribe
ningún artefacto de la cohorte.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOUR = 3600.0

# --- cohorte congelada y exposición en la ventana --------------------------
cohort = pd.read_csv(ROOT / "data/04_multimodal_abx/cohort.csv",
                     usecols=["stay_id", "label", "split"])
treat = pd.read_csv(ROOT / "data/04_multimodal_abx/treatments_timeseries.csv",
                    usecols=["stay_id", "concept"])
exposed = set(treat.loc[treat.concept == "antibiotic_active", "stay_id"].astype(int))
cohort["abx_in_window"] = cohort.stay_id.astype(int).isin(exposed)

# --- tiempos del antibiótico de la sospecha (solo positivos) ---------------
labels = pd.read_csv(ROOT / "data/03_labels/sepsis3_onset.csv", sep="|",
                     usecols=["stay_id", "antibiotic_time", "onset_time", "prediction_time"])
labels["abx_minus_t0_h"] = (labels.antibiotic_time - labels.prediction_time) / HOUR
labels["onset_minus_abx_h"] = (labels.onset_time - labels.antibiotic_time) / HOUR
out = cohort.merge(labels[["stay_id", "abx_minus_t0_h", "onset_minus_abx_h"]],
                   on="stay_id", how="left")
out.loc[out.label == 0, ["abx_minus_t0_h", "onset_minus_abx_h"]] = float("nan")
assert len(out) == len(cohort) == out.stay_id.nunique()
assert out.loc[out.label == 1, "abx_minus_t0_h"].notna().all()

# --- resumen por partición y brazo ------------------------------------------
rows = []
for split in ["train", "val", "test", "all"]:
    part = out if split == "all" else out[out.split == split]
    for label, arm in [(0, "control"), (1, "positive")]:
        g = part[part.label == label]
        row = {"split": split, "arm": arm, "n": len(g),
               "n_abx_in_window": int(g.abx_in_window.sum()),
               "pct_abx_in_window": round(100 * g.abx_in_window.mean(), 2)}
        if label == 1:
            d, o = g.abx_minus_t0_h, g.onset_minus_abx_h
            row.update({
                "pct_abx_after_t0": round(100 * (d > 0).mean(), 2),
                "median_abx_minus_t0_h": round(d.median(), 2),
                "p25_abx_minus_t0_h": round(d.quantile(0.25), 2),
                "p75_abx_minus_t0_h": round(d.quantile(0.75), 2),
                "pct_onset_before_abx": round(100 * (o < 0).mean(), 2),
                "median_onset_minus_abx_h": round(o.median(), 2),
            })
        rows.append(row)
summary = pd.DataFrame(rows)

print(summary.to_string(index=False))
dest = ROOT / "data/05_results/antibiotic_exposure.csv"
out.to_csv(dest, index=False, float_format="%.17g")
summary.to_csv(ROOT / "data/05_results/antibiotic_exposure_summary.csv", index=False)
print(f"escrito {dest.name} ({len(out)} filas) y antibiotic_exposure_summary.csv")
