# TFM — Predicción temprana de sepsis en UCI (MIMIC-IV v3.1)

Trabajo Fin de Máster: predicción temprana de sepsis (criterios Sepsis-3) en
pacientes de UCI, con 6 horas de antelación al *onset*, a partir de datos
estructurados multimodales de [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/).
Este README describe el estado real y actual del proyecto.

## Qué hace este repositorio

1. **Pipeline de datos** (`multi_modality_code/`, 6 etapas, las mismas que
   enumera §5.2 de la memoria, donde está el detalle metodológico completo): extracción desde MIMIC-IV v3.1, etiquetado
   Sepsis-3 (infección sospechada + SOFA ≥ 2), exportación multimodal
   estructurada (estático, signos vitales, laboratorio, tratamientos) con
   ventanas temporales sin fuga de información, validación automática de
   integridad, cálculo de las escalas clínicas SOFA y qSOFA en el instante de
   predicción —que sirven de comparador— y el ciclo de experimentos.
   `2_audit_inputs.py` es una utilidad de auditoría de las entradas, fuera de
   esas seis etapas: ninguna posterior depende de él.
2. **Baselines clásicos de ML** (`multi_modality_code/experiments/`):
   regresión logística, HistGradientBoosting y XGBoost, con 8 configuraciones
   de ablación por modalidad, ajuste de umbral en validación, intervalos de
   confianza bootstrap e importancia de variables por permutación. **Ya
   ejecutados**; resultados en `data/05_results/`.
3. **Modelo de aprendizaje profundo** (`multi_modality_code/experiments/dl/`):
   GRU/LSTM con codificadores independientes por modalidad y fusión
   intermedia (concatenación de representaciones antes de una cabecera de
   clasificación compartida), sobre una representación en rejilla horaria de
   las mismas modalidades y ablaciones. **Ya ejecutado**: 8 ablaciones × 3
   semillas, con intervalos de confianza bootstrap, media ± desviación entre
   semillas e importancia de variables por permutación sobre la ablación
   principal; resultados en `data/05_results/deep_learning/`. Ver
   `multi_modality_code/experiments/dl/README.md` para los comandos y la
   sección de reproducibilidad.

## Estructura del repositorio

```
multi_modality_code/
    1_preprocess_mimic.py          # Etapa 1: extracción desde MIMIC-IV v3.1
    2_audit_inputs.py              # Etapa 1b: auditoría/QC de las tablas extraídas (solo lectura)
    3_build_sepsis3_labels.py      # Etapa 2: etiquetado Sepsis-3 (SOFA horario + regla)
    4_build_multimodal_dataset.py  # Etapa 3: exportación multimodal (data/04_multimodal/)
    5_validate_sepsis3_labels.py   # Etapa 4: validación de integridad/fuga
    6_build_clinical_scores.py     # Etapa 6: SOFA y qSOFA en t0 (comparador clínico)
    experiments/
        run_experiments.py         # Etapa 5: baselines clásicos + ablaciones + CIs + importancia
        evaluate.py                # Métricas, calibración, bootstrap_ci(), ECE, VPP por prevalencia
        posthoc_analysis.py        # Pruebas pareadas entre modelos + reglas clínicas de corte
        threshold_report.py        # Comparación de criterios de umbral (F1 vs Youden)
        data_utils/splits.py       # Partición train/val/test agrupada por paciente
        features/aggregate.py      # Agregación tabular por modalidad
        models/baselines.py        # Regresión logística, HistGBT, XGBoost
        dl/                        # Modelo GRU/LSTM de fusión intermedia (ver dl/README.md)
        README.md                  # Runbook detallado de la capa de experimentos
tests/                             # Suite pytest: leakage, splits, esquema de agregación
data/05_results/                   # CSV y JSON agregados de resultados (lo único de data/ versionado)
pyproject.toml                     # Dependencias y metadatos del proyecto (fuente única)
uv.lock                            # Versiones exactas resueltas (reproducibilidad)
requirements.txt                   # Generado con `uv export`; no editar a mano
```

> **El prefijo numérico del fichero no es el número de etapa.** El prefijo ordena los
> scripts en el listado del directorio; la numeración de etapas es la del pipeline y la
> que usan la memoria y `workflow_script_execution.md`. Divergen porque `2_audit_inputs.py`
> es una comprobación opcional de solo lectura (etapa 1b) y porque la etapa 5 vive en
> `experiments/`, no en un script numerado. La correspondencia completa está en la tabla
> resumen de
> [`workflow_script_execution.md`](multi_modality_code/workflow_script_execution.md).

## Cómo reproducir

Requiere acceso credencializado a MIMIC-IV v3.1 en PhysioNet (curso de ética +
acuerdo de uso de datos).

El entorno se gestiona con [uv](https://docs.astral.sh/uv/) y Python 3.11; `uv sync`
crea el entorno virtual e instala el proyecto como paquete, de modo que los scripts
del pipeline se ejecutan desde la raíz del repositorio sin necesidad de `PYTHONPATH`.

```bash
uv sync                      # crea .venv (Python 3.11) desde pyproject.toml + uv.lock

# Etapa 1: extracción desde MIMIC-IV crudo (la única que necesita los 91 GB originales)
uv run python multi_modality_code/1_preprocess_mimic.py ...
uv run python multi_modality_code/2_audit_inputs.py              # etapa 1b, opcional

# Etapas 2-4: etiquetado, exportación multimodal y validación
uv run python multi_modality_code/3_build_sepsis3_labels.py --lead_time 6 --control_ratio 1.0
uv run python multi_modality_code/4_build_multimodal_dataset.py --output_dir data/04_multimodal
uv run python multi_modality_code/4_build_multimodal_dataset.py \
  --output_dir data/04_multimodal_abx --include_antibiotics     # brazo de sensibilidad
uv run python multi_modality_code/5_validate_sepsis3_labels.py   # gates de etiquetado, exit 0

# Etapa 6: SOFA y qSOFA en t0 (comparador clínico), ~1 min
uv run python multi_modality_code/6_build_clinical_scores.py

# Etapa 5: baselines clásicos + ablaciones + CIs + importancia de variables (~13 min)
uv run python -m multi_modality_code.experiments.run_experiments \
  --multimodal_dir data/04_multimodal \
  --abx_dir data/04_multimodal_abx \
  --output_dir data/05_results --seeds 42,43,44

# Brazo profundo: 8 ablaciones × 3 semillas (~7 min en MPS)
uv run python -m multi_modality_code.experiments.dl.run_dl_experiments \
  --output_dir data/05_results/deep_learning --seeds 42,43,44

# Análisis post hoc: no reentrenan, releen las probabilidades guardadas
uv run python -m multi_modality_code.experiments.threshold_report    # F1 frente a Youden
uv run python -m multi_modality_code.experiments.posthoc_analysis    # pareadas + reglas clínicas

# Suite de tests (leakage, splits, esquema de agregación, escalas clínicas)
uv run pytest tests/ -v
```

Dos atajos para no repetir ejecuciones grandes cuando solo cambia el bloque estático:
`1_preprocess_mimic.py --only_demog` (~3 s, no reextrae `chartevents`) y
`4_build_multimodal_dataset.py --skip_timeseries`.

Ver [`multi_modality_code/workflow_script_execution.md`](multi_modality_code/workflow_script_execution.md)
para los argumentos completos de cada etapa y
[`multi_modality_code/experiments/dl/README.md`](multi_modality_code/experiments/dl/README.md)
para el detalle del brazo profundo.

# 


