from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


RESULT_ROOT = Path("simulation/results/5-25-2026-A")
DEFAULT_OUTPUT = RESULT_ROOT / "kedrl_tuning_evaluation_report.pdf"


def fmt(x: object, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        y = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(y):
        return ""
    if abs(y) >= 1000 or (abs(y) > 0 and abs(y) < 1e-3):
        return f"{y:.{digits}e}"
    return f"{y:.{digits}g}"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Paragraph(text, style)


def load_params_text(run_dir: Path) -> str:
    path = run_dir / "params.yaml"
    if not path.exists():
        return "params.yaml was not copied for this configuration."
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keep_prefixes = (
        "n_ids:",
        "n_timepoints:",
        "Z_sim:",
        "experiment:",
        "benchmark:",
        "target_set:",
        "num_grid_points:",
        "lambda_B:",
        "lambda_reg:",
        "kernel:",
        "optimization:",
        "policy:",
    )
    out: list[str] = []
    capture = False
    for line in lines:
        if line and not line.startswith(" "):
            capture = line.startswith(keep_prefixes)
        if capture:
            out.append(line)
    return "\n".join(out[:140])


def completion_status(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in summary.sort_values("combo_id").iterrows():
        run_rel = str(row.get("run_dir", "")).replace("runs/", "")
        run_dir = root / run_rel
        metrics = run_dir / "metrics"
        plots = run_dir / "plots"
        per_run = metrics / "per_run_metrics.csv"
        n_rows = int(pd.read_csv(per_run).shape[0]) if per_run.exists() else 0
        bench_ids = 0
        rep_ids = 0
        if per_run.exists():
            df = pd.read_csv(per_run, usecols=["offline_data_id", "benchmark_id"])
            bench_ids = int(df["benchmark_id"].nunique())
            rep_ids = int(df["offline_data_id"].nunique())
        rows.append(
            {
                "combo_id": int(row["combo_id"]),
                "combo_name": row["combo_name"],
                "metrics_dir": metrics.exists(),
                "plots_dir": plots.exists(),
                "per_run_rows": n_rows,
                "offline_reps_seen": rep_ids,
                "eval_targets_seen": bench_ids,
                "summary_reps": int(row.get("risk_n_replicates", 0)),
            }
        )
    return pd.DataFrame(rows)


def df_table(
    df: pd.DataFrame,
    columns: Iterable[str],
    labels: Iterable[str],
    *,
    max_rows: int | None = None,
    font_size: int = 7,
) -> Table:
    if max_rows is not None:
        df = df.head(max_rows)
    data = [[p(str(label), TABLE_HEADER) for label in labels]]
    for _, row in df.iterrows():
        data.append([p(fmt(row.get(col)), TABLE_CELL) for col in columns])
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c2d0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def scaled_image(path: Path, max_w: float, max_h: float) -> Image:
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def per_benchmark_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics" / "per_run_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = [
        "RMSE",
        "MAE",
        "Bias",
        "Corr",
        "projected_bellman_test_risk",
        "benchmark_embedding_risk",
    ]
    agg = df.groupby("benchmark_id")[cols].agg(["count", "mean", "median", "std"])
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    return agg.reset_index()


def make_report(root: Path, output: Path) -> None:
    summary_path = root / "tuning_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing tuning summary: {summary_path}")
    summary = pd.read_csv(summary_path).sort_values("score").reset_index(drop=True)
    best = summary.iloc[0]
    best_run = root / str(best["run_dir"]).replace("runs/", "")
    base_row = summary.loc[summary["combo_name"].eq("base")]
    base_run = root / str(base_row.iloc[0]["run_dir"]).replace("runs/", "") if not base_row.empty else None
    comp = completion_status(root, summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=landscape(letter),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
        title="KE-DRL tuning evaluation 5-25-2026-A",
    )

    story: list[object] = []
    story.append(p("KE-DRL Tuning Evaluation: 5-25-2026-A", TITLE))
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        p(
            "Verdict: the best copied configuration is combo 5, kernel_sigma_0.7. "
            "It has the lowest composite score, lowest RMSE/MAE, and lowest projected Bellman diagnostic among the seven tested settings. "
            "The improvement is large, but the calibration panel still shows systematic underestimation for several evaluation target points; this is the best current candidate, not yet a perfect final result.",
            BODY,
        )
    )
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        p(
            "Important copy-status caveat: this folder was still being copied from the cluster. "
            "The top-level tuning_summary.csv is present and usable, but some per-config folders are incomplete locally. "
            "The report marks incomplete folders explicitly.",
            BODY,
        )
    )
    story.append(Spacer(1, 0.14 * inch))

    rank_cols = [
        "combo_id",
        "combo_name",
        "score",
        "RMSE_mean",
        "MAE_mean",
        "Bias_mean",
        "Corr_mean",
        "projected_bellman_test_risk_mean",
        "target_mass_rmse_to_target_mean",
        "risk_n_replicates",
    ]
    rank_labels = [
        "id",
        "config",
        "score",
        "RMSE mean",
        "MAE mean",
        "Bias mean",
        "Corr mean",
        "Projected Bellman",
        "mass RMSE",
        "risk reps",
    ]
    story.append(p("Ranked Configuration Summary", H1))
    story.append(df_table(summary, rank_cols, rank_labels))
    story.append(Spacer(1, 0.12 * inch))

    story.append(p("Local Folder Completion", H1))
    story.append(
        df_table(
            comp,
            [
                "combo_id",
                "combo_name",
                "metrics_dir",
                "plots_dir",
                "per_run_rows",
                "offline_reps_seen",
                "eval_targets_seen",
                "summary_reps",
            ],
            [
                "id",
                "config",
                "metrics",
                "plots",
                "per-run rows",
                "reps seen",
                "targets seen",
                "summary reps",
            ],
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        p(
            "Interpretation: sigma=0.7 reduces RMSE from 1.58 to 0.469 versus baseline, MAE from 1.414 to 0.421, "
            "and projected Bellman risk from 5.22 to 0.495. The length-scale 1.5 setting is unusable here: it expands RMSE to 9.61 and has negative average correlation.",
            BODY,
        )
    )

    story.append(PageBreak())
    story.append(p("Best Configuration Figure", H1))
    best_fig = best_run / "plots" / "mu_summary_benchmarks.png"
    if best_fig.exists():
        story.append(scaled_image(best_fig, 10.2 * inch, 6.4 * inch))
    else:
        story.append(p(f"Missing figure: {best_fig}", BODY))

    if base_run is not None:
        story.append(PageBreak())
        story.append(p("Baseline Figure", H1))
        base_fig = base_run / "plots" / "mu_summary_benchmarks.png"
        if base_fig.exists():
            story.append(scaled_image(base_fig, 10.2 * inch, 6.4 * inch))
        else:
            story.append(p(f"Missing figure: {base_fig}", BODY))

    story.append(PageBreak())
    story.append(p("Per Evaluation Target: Best Configuration", H1))
    bench = per_benchmark_summary(best_run)
    if not bench.empty:
        story.append(
            df_table(
                bench,
                [
                    "benchmark_id",
                    "RMSE_count",
                    "RMSE_mean",
                    "RMSE_median",
                    "MAE_mean",
                    "MAE_median",
                    "Bias_mean",
                    "Corr_mean",
                    "projected_bellman_test_risk_mean",
                    "projected_bellman_test_risk_median",
                    "benchmark_embedding_risk_mean",
                ],
                [
                    "target",
                    "n",
                    "RMSE mean",
                    "RMSE med",
                    "MAE mean",
                    "MAE med",
                    "Bias mean",
                    "Corr mean",
                    "Bellman mean",
                    "Bellman med",
                    "embed risk mean",
                ],
                font_size=6,
            )
        )
    else:
        story.append(p("The best configuration per-run metrics were not fully copied.", BODY))

    story.append(Spacer(1, 0.12 * inch))
    story.append(p("Scientific Reading", H1))
    bullets = [
        "The best setting is clearly kernel sigma=0.7, with baseline regularization lambda_B=0.02, nu=5.5, length_scale=1.0.",
        "The result is satisfactory as a tuning-screen result because the ranking is decisive and the support-safe evaluation targets no longer produce a trivial zero curve.",
        "The result is not yet satisfactory as the final reportable simulation: calibration is compressed toward zero and several target points remain biased downward.",
        "Evaluation target point 5 is easy across all configurations; points 1, 2, and 3 are the hard cases and should dominate the final narrative.",
        "The final run should therefore use sigma=0.7 but increase Monte Carlo precision, offline replications, and sample size, then regenerate the combined box/diagnostic figures from the final one-config run.",
    ]
    for item in bullets:
        story.append(p(f"- {item}", BODY))

    story.append(Spacer(1, 0.12 * inch))
    story.append(p("Best Configuration Parameters Found in Copied Run", H1))
    story.append(p(load_params_text(best_run).replace("\n", "<br/>"), MONO))

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(RESULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    make_report(Path(args.root), Path(args.output))
    print(f"Wrote {Path(args.output).resolve()}")


styles = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    fontSize=20,
    leading=24,
    alignment=TA_LEFT,
    spaceAfter=6,
)
H1 = ParagraphStyle(
    "HeadingCustom",
    parent=styles["Heading2"],
    fontSize=13,
    leading=15,
    spaceBefore=5,
    spaceAfter=4,
)
BODY = ParagraphStyle(
    "BodyCustom",
    parent=styles["BodyText"],
    fontSize=9,
    leading=12,
)
MONO = ParagraphStyle(
    "MonoCustom",
    parent=styles["Code"],
    fontName="Courier",
    fontSize=6,
    leading=7,
)
TABLE_HEADER = ParagraphStyle(
    "TableHeader",
    parent=styles["BodyText"],
    fontSize=6.5,
    leading=8,
)
TABLE_CELL = ParagraphStyle(
    "TableCell",
    parent=styles["BodyText"],
    fontSize=6.2,
    leading=7.2,
)


if __name__ == "__main__":
    main()
