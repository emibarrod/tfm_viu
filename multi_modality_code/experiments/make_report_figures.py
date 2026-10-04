"""Build the figures the thesis report uses for the results chapter.

Nothing is retrained here. Point estimates and confidence intervals are read
from the canonical CSVs (`metrics.csv`, `deep_learning/dl_metrics.csv`) so that
every figure agrees with the printed tables by construction, and the predicted
probabilities persisted in `predictions/*.npz` are used only where a curve needs
the raw scores (ROC, precision-recall, calibration).

Each builder asserts its plotted values against the canonical CSV before saving,
so a figure that has drifted away from the tables fails loudly instead of
shipping. Run with:

    uv run python multi_modality_code/experiments/make_report_figures.py
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from multi_modality_code.experiments.evaluate import expected_calibration_error, ppv_at_prevalence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "05_results")
DL_DIR = os.path.join(RESULTS_DIR, "deep_learning")
FIGURES_DIR = os.path.join(REPO_ROOT, "tfm_latex", "Images", "resultados")

PRIMARY_ABLATION = "static_plus_vitals_plus_labs_plus_treatments"

# Display order and labels for the four models. The three classical baselines
# live in metrics.csv; the GRU lives in dl_metrics.csv under its own model name.
# Two label forms, both taken from the chapter's tables: the long one for the
# single-panel figures, the abbreviated one where a legend has to fit a half-width
# panel.
MODELS = [
    ("logreg", "Regresión logística", "Log. Reg.", "#4C72B0", "o", "-"),
    ("tree_hgbt", "HistGradientBoosting", "HistGBT", "#C44E52", "s", "-"),
    ("xgboost", "XGBoost", "XGBoost", "#DD8452", "^", "--"),
    ("gru_intermediate_fusion", "GRU (fusión intermedia)", "GRU", "#55A868", "D", "-."),
]

# The eight modality ablations, bottom to top, as the ladder is read: single
# modalities first, then the combinations, then the leaky sensitivity run.
ABLATION_ORDER = [
    ("labs_only", "Laboratorio"),
    ("treatments_no_antibiotics", "Tratamientos"),
    ("static_only", "Estático"),
    ("vitals_only", "Vitales"),
    ("vitals_plus_labs", "Vitales + Labs"),
    ("static_plus_vitals_plus_labs", "Est. + Vit. + Labs"),
    (PRIMARY_ABLATION, "Completa (principal)"),
    ("sensitivity_with_antibiotics", "Completa + antibióticos"),
]

# Prevalences the operational reading singles out, and the cohort's own.
REFERENCE_PREVALENCES = (0.05, 0.10)
COHORT_PREVALENCE = 0.5

PLOT_STYLE = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "savefig.dpi": 200,
    "figure.constrained_layout.use": True,
}


def comma(value: float, decimals: int = 3) -> str:
    """Format a number the way the thesis does, with a decimal comma."""
    return f"{value:.{decimals}f}".replace(".", ",")


def comma_axis(*axes, decimals: int = 2) -> None:
    """Put decimal commas on the tick labels, as everywhere else in the memoir."""
    formatter = matplotlib.ticker.FuncFormatter(lambda v, _: comma(v, decimals))
    for axis in axes:
        axis.set_major_formatter(formatter)


def load_metrics() -> pd.DataFrame:
    """Both arms' metrics in one frame, keyed by (ablation, model)."""
    classical = pd.read_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
    deep = pd.read_csv(os.path.join(DL_DIR, "dl_metrics.csv"))
    shared = [c for c in classical.columns if c in deep.columns]
    return pd.concat([classical[shared], deep[shared]], ignore_index=True)


def predictions_path(ablation: str, model: str, seed: int) -> str:
    """Locate the persisted probabilities for one run."""
    if model == "gru_intermediate_fusion":
        return os.path.join(DL_DIR, "predictions", f"{ablation}__gru_seed{seed}.npz")
    return os.path.join(RESULTS_DIR, "predictions", f"{ablation}__{model}_seed{seed}.npz")


def load_test_predictions(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Test labels and probabilities of a row's median-seed run."""
    path = predictions_path(row["ablation"], row["model"], int(row["median_seed"]))
    with np.load(path) as data:
        return data["y_test"].astype(int), data["test_prob"].astype(float)


def figure_ablation_ladder(metrics: pd.DataFrame) -> str:
    """AUROC with its 95 % CI for every modality ablation and model."""
    output = os.path.join(FIGURES_DIR, "ablaciones_auroc.png")
    offsets = np.linspace(0.28, -0.28, len(MODELS))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for (model, label, _, color, marker, _style), offset in zip(MODELS, offsets):
            xs, ys, lows, highs = [], [], [], []
            for index, (ablation, _) in enumerate(ABLATION_ORDER):
                row = metrics[(metrics["ablation"] == ablation) & (metrics["model"] == model)]
                if row.empty:  # clinical_scores_only and the GRU's missing runs
                    continue
                row = row.iloc[0]
                # Both tables report the median-seed run, which is what `auroc`
                # holds; the GRU's mean over seeds is a separate column.
                assert row["auroc_ci_low"] <= row["auroc"] <= row["auroc_ci_high"], (ablation, model)
                xs.append(row["auroc"])
                lows.append(row["auroc"] - row["auroc_ci_low"])
                highs.append(row["auroc_ci_high"] - row["auroc"])
                ys.append(index + offset)
            ax.errorbar(
                xs, ys, xerr=[lows, highs], fmt=marker, color=color, label=label,
                markersize=4, linewidth=0, elinewidth=1.1, capsize=2.2,
            )

        ax.axvline(0.5, color="0.4", linestyle=":", linewidth=1)
        ax.text(0.503, -0.45, "azar", color="0.35", fontsize=7.5, va="center")

        # The leaky run sits apart from the seven legitimate configurations.
        separator = len(ABLATION_ORDER) - 1.5
        ax.axhline(separator, color="0.75", linewidth=0.8)
        ax.text(0.512, separator + 0.12, "con fuga de etiqueta", color="0.35", fontsize=7,
                va="bottom")

        ax.set_yticks(range(len(ABLATION_ORDER)))
        ax.set_yticklabels([label for _, label in ABLATION_ORDER])
        ax.set_xlabel("AUROC en test [IC 95 %]")
        ax.set_xlim(0.46, 0.83)
        ax.set_ylim(-0.7, len(ABLATION_ORDER) - 0.4)
        comma_axis(ax.xaxis)
        ax.legend(loc="lower right", framealpha=0.95)
        fig.savefig(output)
        plt.close(fig)
    return output


def figure_ppv_vs_prevalence(metrics: pd.DataFrame) -> str:
    """PPV of the primary operating point as prevalence moves from 0 to 50 %."""
    output = os.path.join(FIGURES_DIR, "vpp_prevalencia.png")
    prevalences = np.linspace(0.005, COHORT_PREVALENCE, 400)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        for model, label, _, color, marker, linestyle in MODELS:
            row = metrics[(metrics["ablation"] == PRIMARY_ABLATION) & (metrics["model"] == model)]
            row = row.iloc[0]
            sensitivity, specificity = row["sensitivity"], row["specificity"]
            curve = [ppv_at_prevalence(sensitivity, specificity, p) for p in prevalences]

            # Guard: the curve must pass through the two values the table prints.
            for prevalence, column in zip(REFERENCE_PREVALENCES,
                                          ["ppv_at_5pct_prevalence", "ppv_at_10pct_prevalence"]):
                assert abs(ppv_at_prevalence(sensitivity, specificity, prevalence)
                           - row[column]) < 1e-9, (model, column)
            assert abs(ppv_at_prevalence(sensitivity, specificity, COHORT_PREVALENCE)
                       - row["ppv"]) < 1e-9, model

            ax.plot(prevalences * 100, curve, color=color, linestyle=linestyle, linewidth=1.6,
                    label=label)
            ax.plot([p * 100 for p in REFERENCE_PREVALENCES],
                    [ppv_at_prevalence(sensitivity, specificity, p) for p in REFERENCE_PREVALENCES],
                    marker, color=color, markersize=4, linestyle="none")

        low, high = (p * 100 for p in REFERENCE_PREVALENCES)
        ax.axvspan(low, high, color="0.85", alpha=0.55, zorder=0)
        for prevalence in (low, high):
            ax.axvline(prevalence, color="0.55", linestyle=":", linewidth=1)
        ax.text(high + 1.5, 0.61, "prevalencias de referencia (5-10 %)", color="0.3",
                fontsize=7.5, va="center")
        ax.annotate("prevalencia de la cohorte\n(caso-control 1:1)",
                    xy=(49.6, 0.69), xytext=(31, 0.86), fontsize=7.5, color="0.3",
                    ha="center", arrowprops=dict(arrowstyle="->", color="0.5", linewidth=0.8))

        ax.set_xlabel("Prevalencia de sepsis en la población (%)")
        ax.set_ylabel("Valor predictivo positivo")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1)
        comma_axis(ax.yaxis)
        ax.legend(loc="lower right", framealpha=0.95)
        fig.savefig(output)
        plt.close(fig)
    return output


def figure_roc_pr(metrics: pd.DataFrame) -> str:
    """ROC and precision-recall curves of the primary configuration."""
    output = os.path.join(FIGURES_DIR, "roc_pr_principal.png")

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(6.6, 3.3))
        for model, _, label, color, _marker, linestyle in MODELS:
            row = metrics[(metrics["ablation"] == PRIMARY_ABLATION)
                          & (metrics["model"] == model)].iloc[0]
            y_true, y_prob = load_test_predictions(row)

            auroc, auprc = roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)
            # Guard: the curves must belong to the run the tables report.
            assert abs(auroc - row["auroc"]) < 1e-9, (model, auroc, row["auroc"])
            assert abs(auprc - row["auprc"]) < 1e-9, (model, auprc, row["auprc"])

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax_roc.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=1.4,
                        label=f"{label} ({comma(auroc)})")

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ax_pr.plot(recall, precision, color=color, linestyle=linestyle, linewidth=1.4,
                       label=f"{label} ({comma(auprc)})")

        prevalence = float(np.mean(y_true))
        ax_roc.plot([0, 1], [0, 1], color="0.4", linestyle=":", linewidth=1)
        ax_roc.text(0.72, 0.66, "azar", color="0.35", fontsize=7, rotation=39,
                    rotation_mode="anchor")
        ax_roc.set_xlabel("1 − especificidad")
        ax_roc.set_ylabel("Sensibilidad")
        ax_roc.set_title("Curva ROC")
        ax_roc.legend(loc="lower right", title="Modelo (AUROC)", title_fontsize=7.5)

        ax_pr.axhline(prevalence, color="0.4", linestyle=":", linewidth=1)
        ax_pr.text(0.025, prevalence + 0.014, f"azar (prevalencia {comma(prevalence, 2)})",
                   color="0.35", fontsize=7, va="bottom")
        ax_pr.set_xlabel("Exhaustividad")
        ax_pr.set_ylabel("Precisión")
        ax_pr.set_title("Curva precisión-exhaustividad")
        ax_pr.set_ylim(0.35, 1.02)
        ax_pr.legend(loc="upper right", title="Modelo (AUPRC)", title_fontsize=7.5)

        for ax in (ax_roc, ax_pr):
            ax.set_xlim(0, 1)
            comma_axis(ax.xaxis, ax.yaxis, decimals=1)
        fig.savefig(output)
        plt.close(fig)
    return output


def figures_calibration(metrics: pd.DataFrame) -> list[str]:
    """Redraw the three calibration curves with Spanish labels.

    The originals come from `save_calibration_plot`, whose axis labels are in
    English; everything else in the chapter is in Spanish. Same data, same ten
    quantile bins, same median-seed run.
    """
    outputs = []
    panels = [("logreg", "logreg"), ("tree_hgbt", "hgbt"), ("gru_intermediate_fusion", "gru")]

    with plt.rc_context(PLOT_STYLE):
        for model, suffix in panels:
            row = metrics[(metrics["ablation"] == PRIMARY_ABLATION)
                          & (metrics["model"] == model)].iloc[0]
            y_true, y_prob = load_test_predictions(row)
            assert abs(expected_calibration_error(y_true, y_prob) - row["ece"]) < 1e-9, model

            label = next(text for key, text, *_ in MODELS if key == model)
            color = next(c for key, _, _, c, _, _ in MODELS if key == model)
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", linewidth=1,
                    label="Calibración perfecta")
            ax.plot(prob_pred, prob_true, marker="o", markersize=3.5, color=color, linewidth=1.3,
                    label=label)
            ax.set_xlabel("Probabilidad predicha")
            ax.set_ylabel("Frecuencia observada")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            comma_axis(ax.xaxis, ax.yaxis, decimals=1)
            ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9)

            output = os.path.join(FIGURES_DIR, f"calibracion_primaria_{suffix}.png")
            fig.savefig(output)
            plt.close(fig)
            outputs.append(output)
    return outputs


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metrics = load_metrics()
    built = [
        figure_ablation_ladder(metrics),
        figure_ppv_vs_prevalence(metrics),
        figure_roc_pr(metrics),
        *figures_calibration(metrics),
    ]
    for path in built:
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
