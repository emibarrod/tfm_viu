"""Análisis de sensibilidad del comparador clínico sobre la cohorte congelada (post hoc).

Pregunta: el hallazgo de §6.4 —que el SOFA y el qSOFA evaluados en `t0` no
discriminan— ¿es un artefacto de algún subgrupo? Se comprueba estratificando por
tres vías y midiendo el AUROC de las escalas dentro de cada estrato:

  1. cuartiles de duración de la estancia en UCI (`los`, de `demog_processed.csv`);
  2. unidad de UCI (1=MICU, 2=SICU, 3=TSICU, 4=CSRU/cardiaca, 5=NeuroInt, 6=CCU,
     0=otra), con la codificación de `1_preprocess_mimic.py:460-465`;
  3. restricción a las estancias cuya ventana de observación completa cae dentro
     de la UCI (`lower_bound >= intime`).

Las cifras son de la **cohorte completa** (9.278 estancias), no del conjunto de
test: estratificar 1.392 estancias en cuartiles deja 348 por celda, demasiado
pocas para que la lectura signifique algo. La memoria lo declara así.

No reentrena nada, no lee ningún modelo y no reescribe ningún artefacto previo:
solo lee la cohorte congelada y las escalas ya calculadas por la etapa
`6_build_clinical_scores.py`, y escribe un CSV nuevo.

    uv run python -m multi_modality_code.experiments.clinical_scores_sensitivity
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MULTIMODAL_DIR = ROOT / "data/04_multimodal"
DEST = ROOT / "data/05_results/clinical_scores_sensitivity.csv"

# `1_preprocess_mimic.py:422`. Solo se nombran las que la memoria cita.
UNIT_NAMES = {0: "otra", 1: "medica_MICU", 2: "quirurgica_SICU", 3: "trauma_TSICU",
              4: "cardiaca_CSRU", 5: "neuro_intermedia", 6: "coronaria_CCU"}
SCORES = ("sofa_total_at_t0", "qsofa_at_t0")
# Un estrato con menos estancias que esto no se reporta: el AUROC sería ruido.
MIN_STRATUM = 100


def load_cohort() -> pd.DataFrame:
    """Cohorte congelada con las escalas en t0, la unidad y la duración de estancia."""
    scores = pd.read_csv(MULTIMODAL_DIR / "clinical_scores.csv", float_precision="round_trip")
    cohort = pd.read_csv(MULTIMODAL_DIR / "cohort.csv", float_precision="round_trip")
    static = pd.read_csv(MULTIMODAL_DIR / "static.csv", float_precision="round_trip")
    demog = pd.read_csv(ROOT / "data/02_onset/demog_processed.csv", sep="|",
                        float_precision="round_trip")
    demog["stay_id"] = pd.to_numeric(demog["stay_id"], errors="coerce")
    # `los` viene en días en la tabla de origen; la cohorte razona en horas.
    demog["los_hours"] = pd.to_numeric(demog["los"], errors="coerce") * 24.0

    return (scores[["stay_id", "label", *SCORES]]
            .merge(cohort[["stay_id", "intime", "lower_bound"]], on="stay_id")
            .merge(static[["stay_id", "unit"]], on="stay_id")
            .merge(demog[["stay_id", "los_hours"]], on="stay_id"))


def row(kind: str, name: str, group: pd.DataFrame) -> dict | None:
    """Una fila del CSV: el AUROC de cada escala dentro de un estrato."""
    if len(group) < MIN_STRATUM or group["label"].nunique() < 2:
        return None
    out = {"stratum_kind": kind, "stratum": name, "n": len(group),
           "n_positive": int(group["label"].sum()),
           "pct_positive": round(100 * float(group["label"].mean()), 2)}
    for score in SCORES:
        out[f"auroc_{score.replace('_at_t0', '')}"] = float(
            roc_auc_score(group["label"], group[score]))
    return out


def main() -> None:
    data = load_cohort()
    rows = [row("cohorte_completa", "todas", data)]

    quartile = pd.qcut(data["los_hours"], 4, labels=False, duplicates="drop")
    for q, group in data.groupby(quartile):
        lo, hi = group["los_hours"].min(), group["los_hours"].max()
        rows.append(row("cuartil_los_uci", f"Q{int(q) + 1} ({lo:.0f}-{hi:.0f} h)", group))

    for unit, group in data.groupby("unit"):
        rows.append(row("unidad_uci", UNIT_NAMES.get(int(unit), str(unit)), group))

    # La ventana [t0-24h, t0] cae entera dentro de la UCI: descarta las estancias
    # cuya ventana empieza antes del ingreso en la unidad.
    rows.append(row("ventana_integra_en_uci", "lower_bound >= intime",
                    data[data["lower_bound"] >= data["intime"]]))

    out = pd.DataFrame([r for r in rows if r is not None])
    out.to_csv(DEST, index=False, float_format="%.17g")

    print(f"{'estrato':<40}{'n':>6}{'% pos':>8}{'SOFA':>8}{'qSOFA':>8}")
    for r in out.itertuples(index=False):
        print(f"{r.stratum_kind + ' / ' + r.stratum:<40}{r.n:>6}{r.pct_positive:>8.1f}"
              f"{r.auroc_sofa_total:>8.3f}{r.auroc_qsofa:>8.3f}")
    print(f"\nescrito {DEST.relative_to(ROOT)} ({len(out)} filas)")


if __name__ == "__main__":
    main()
