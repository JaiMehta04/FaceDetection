"""
Streamlit Dashboard — Face Detection & Anonymization Analysis

A fully self-contained, deployable dashboard that reads CSV outputs and
pre-generated plots, then renders an interactive report with:
  - KPI cards for each detector variant
  - Side-by-side metrics comparison (interactive Plotly charts)
  - Group-wise performance degradation
  - Quality analysis with explanations of every metric
  - Distance-based failure analysis
  - Pre-rendered plot gallery
  - Full text report viewer

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "outputs" / "csv"
PLOTS_DIR = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs" / "reports"
EXP_CSV_DIR = ROOT / "experiments" / "csv"

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdaSR-Face — Detection & Super-Resolution Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    [data-testid="stSidebar"] { background: #0f1117; }
    [data-testid="stSidebar"] * { color: #c8cdd3 !important; }
    [data-testid="stSidebar"] .stRadio label:hover { color: #fff !important; }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 14px; padding: 32px 36px; color: #e0e0e0;
        margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.06);
    }
    .hero-banner h1 { color: #fff; margin: 0 0 6px 0; font-size: 28px; }
    .hero-banner p  { margin: 0; font-size: 15px; line-height: 1.6; opacity: 0.88; }
    .hero-banner .hero-tag {
        display: inline-block; background: rgba(46,204,113,0.18);
        border: 1px solid rgba(46,204,113,0.35); border-radius: 6px;
        padding: 2px 10px; font-size: 12px; color: #2ecc71; margin-top: 10px;
    }

    /* ── Gradient metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin: 5px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card h1 { margin: 6px 0 0 0; font-size: 34px; font-weight: 800; }
    .metric-green  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
    .metric-red    { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .metric-blue   { background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%); }

    /* ── Info / callout boxes ── */
    .info-box {
        background-color: #f0f4ff; border-left: 4px solid #2196F3;
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
        font-size: 13.5px; color: #333; line-height: 1.65;
    }
    .callout-success {
        background: #e8f8f0; border-left: 4px solid #2ecc71;
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
        font-size: 13.5px; color: #1a5632; line-height: 1.65;
    }
    .callout-warn {
        background: #fff8e6; border-left: 4px solid #f39c12;
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
        font-size: 13.5px; color: #7d5a00; line-height: 1.65;
    }

    /* ── Section divider ── */
    .section-divider { border-top: 2px solid #eee; margin: 30px 0 20px 0; }

    /* ── Sidebar section headers ── */
    .sidebar-section {
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.2px; color: #667 !important; margin: 18px 0 4px 0;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def discover_variants():
    """Find all detector variants from metrics CSVs."""
    variants = {}
    for f in sorted(CSV_DIR.glob("metrics_*.csv")):
        name = f.stem.replace("metrics_", "")
        variants[name] = {
            "metrics": load_csv(f),
            "quality": load_csv(CSV_DIR / f"quality_{name}.csv"),
            "quality_summary": load_csv(CSV_DIR / f"quality_summary_{name}.csv"),
            "groupwise": load_csv(CSV_DIR / f"groupwise_{name}.csv"),
            "distance": load_csv(CSV_DIR / f"distance_{name}.csv"),
            "distance_summary": load_csv(CSV_DIR / f"distance_summary_{name}.csv"),
            "attribute": load_csv(CSV_DIR / f"attribute_{name}.csv"),
            "event": load_csv(CSV_DIR / f"event_{name}.csv"),
            "pr_curve": load_csv(CSV_DIR / f"pr_curve_{name}.csv"),
            "threshold": load_csv(CSV_DIR / f"threshold_{name}.csv"),
        }
    return variants


@st.cache_data
def load_report_text():
    """Load the latest text report."""
    reports = sorted(REPORTS_DIR.glob("analysis_report_*.txt"))
    if reports:
        return reports[-1].read_text(encoding="utf-8"), reports[-1].name
    return "", "No report found"


def display_name(variant: str) -> str:
    """Make variant names human-readable."""
    return (variant
            .replace("_insightface_det_", " — ")
            .replace("_insightface", "")
            .replace("_", " "))


# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
variants = discover_variants()

if not variants:
    st.error("No CSV data found in `outputs/csv/`. Run the pipeline first: `python main.py`")
    st.stop()

variant_names = list(variants.keys())

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 AdaSR-Face")
    st.caption("Detection & Super-Resolution Research")

    st.markdown("---")
    st.markdown('<p style="font-size:11px;font-weight:700;text-transform:uppercase;'
                'letter-spacing:1.2px;color:#888;margin:0 0 2px 0;">Overview</p>',
                unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Detector Showdown",
            "🔬 Why Faces Are Missed",
            "🌆 Scene & Crowd Analysis",
            "🧬 AdaSR Ablation Study",
            "🧠 Conclusions & Next Steps",
            "📎 Appendix",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p style="font-size:11px;font-weight:700;text-transform:uppercase;'
                'letter-spacing:1.2px;color:#888;margin:0 0 4px 0;">Dataset</p>',
                unsafe_allow_html=True)
    st.caption("WIDER FACE Validation · 3,226 images · 39,123 faces")

    st.markdown('<p style="font-size:11px;font-weight:700;text-transform:uppercase;'
                'letter-spacing:1.2px;color:#888;margin:12px 0 4px 0;">Detectors</p>',
                unsafe_allow_html=True)
    for v in variant_names:
        st.caption(f"• {display_name(v)}")

    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · ONNX Runtime")



# ══════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <h1>🧬 AdaSR-Face: Adaptive Super-Resolution for Face Detection</h1>
        <p>
            This dashboard presents a comprehensive evaluation of face detection models
            on the <b>WIDER FACE</b> benchmark (3,226 images, 39,123 annotated faces),
            plus an ablation study for <b>AdaSR-Face</b> — a novel pipeline that uses
            <b>confidence-guided selective super-resolution</b> to recover small faces
            that standard detectors miss.
        </p>
        <span class="hero-tag">Research Dashboard · WIDER FACE Val · RetinaFace · MTCNN · SCRFD · Real-ESRGAN</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-look KPIs ──
    all_metrics = []
    for v, data in variants.items():
        row = data["metrics"].iloc[0].to_dict()
        row["variant"] = v
        all_metrics.append(row)
    home_df = pd.DataFrame(all_metrics)
    home_best = home_df.loc[home_df["f1"].astype(float).idxmax()]

    st.markdown("### At a Glance")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Best Detector F1", f"{float(home_best['f1']):.3f}")
    h2.metric("Best Recall", f"{float(home_best['recall']):.3f}")
    h3.metric("Best Precision", f"{float(home_best['precision']):.3f}")
    h4.metric("Detector Variants", f"{len(variants)}")
    h5.metric("Total GT Faces", f"{int(home_best['total_gt']):,}")

    st.markdown("---")

    # ── Roadmap cards ──
    st.markdown("### 🗺️ What's Inside — Dashboard Guide")
    st.markdown("""
    <div class="info-box">
    This dashboard has <b>6 sections</b>. Here's what each one answers and why it matters:
    </div>
    """, unsafe_allow_html=True)

    guide = [
        ("📊 Detector Showdown",
         "Which detector wins?",
         "Side-by-side comparison of RetinaFace, MTCNN, and their enhanced variants "
         "(tiled inference, multi-scale). Includes PR curves, threshold tuning, and "
         "crowd-density impact.",
         "#2196F3"),
        ("🔬 Why Faces Are Missed",
         "What causes detection failures?",
         "Deep-dive into the 11,000+ missed faces: face size distributions, brightness, "
         "blur, occlusion, proximity analysis, and per-attribute recall breakdowns.",
         "#e74c3c"),
        ("🌆 Scene & Crowd Analysis",
         "Which scenes are hardest?",
         "Performance across 61 WIDER FACE event categories (Parade, Basketball, Meeting…). "
         "Shows which real-world scenarios break the detector.",
         "#FF9800"),
        ("🧬 AdaSR Ablation Study",
         "Can super-resolution fix small-face misses?",
         "The core research contribution. Compares SCRFD baseline, bicubic upscaling, "
         "Real-ESRGAN blind SR, and our novel <b>AdaSR-Face</b> selective SR pipeline. "
         "Includes per-size-bucket recall heatmaps showing exactly where SR helps.",
         "#9b59b6"),
        ("🧠 Conclusions & Next Steps",
         "What did we learn?",
         "Data-driven summary: best detector, root causes of failures, "
         "actionable recommendations, and the research roadmap.",
         "#2ecc71"),
        ("📎 Appendix",
         "Raw plots & full report",
         "Pre-rendered matplotlib plots and the complete text analysis report "
         "for reference and paper figures.",
         "#607D8B"),
    ]

    for i in range(0, len(guide), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(guide):
                title, question, desc, color = guide[i + j]
                col.markdown(f"""
                <div style="border-left:5px solid {color}; padding:14px 18px; margin:6px 0;
                            background:#fafafa; border-radius:0 10px 10px 0; min-height:130px;">
                    <b style="font-size:15px;">{title}</b><br>
                    <span style="color:{color}; font-size:14px; font-weight:600;">{question}</span><br>
                    <span style="font-size:13px; color:#555; line-height:1.55;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔑 Key Finding")
    st.markdown("""
    <div class="callout-success">
        <b>94% of all missed faces are smaller than 32×32 pixels.</b> Face size is the dominant
        failure factor across all detectors. This motivates the AdaSR-Face approach: selectively
        super-resolve only the image regions where small faces are likely hiding, guided by
        detection confidence scores.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Detector Showdown
# ══════════════════════════════════════════════════════════════
elif page == "📊 Detector Showdown":
    st.title("📊 Detector Showdown — Performance Comparison")
    st.markdown("Comparing all detector variants evaluated on the **WIDER FACE** validation set.")

    # ── Metric definitions ──
    with st.expander("ℹ️ What do these metrics mean?", expanded=False):
        st.markdown("""
| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Precision** | TP / (TP + FP) | Of all boxes the model drew, how many actually contain a face? |
| **Recall** | TP / (TP + FN) | Of all real faces in the image, how many did the model find? |
| **F1 Score** | 2 × P × R / (P + R) | Harmonic mean of Precision & Recall — balances both |
| **Mean IoU** | avg(Intersection / Union) | How tightly do predicted boxes overlap with ground truth? (1.0 = perfect) |
| **TP** | True Positives | Correctly detected faces (IoU > 0.5 with a GT box) |
| **FP** | False Positives | Predicted boxes that don't match any real face |
| **FN** | False Negatives | Real faces the model completely missed |
        """)

    # ── KPI cards for best detector ──
    all_metrics = []
    for v, data in variants.items():
        row = data["metrics"].iloc[0].to_dict()
        row["variant"] = v
        all_metrics.append(row)
    comp_df = pd.DataFrame(all_metrics)

    best_idx = comp_df["f1"].astype(float).idxmax()
    best = comp_df.iloc[best_idx]

    st.markdown(f"### 🏆 Best Performer: **{display_name(best['variant'])}**")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Precision", f"{float(best['precision']):.3f}")
    c2.metric("Recall", f"{float(best['recall']):.3f}")
    c3.metric("F1 Score", f"{float(best['f1']):.3f}")
    c4.metric("Mean IoU", f"{float(best['mean_iou']):.3f}")
    c5.metric("True Positives", f"{int(best['tp']):,}")
    c6.metric("Total GT Faces", f"{int(best['total_gt']):,}")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Comparison bar chart ──
    st.subheader("Metrics Comparison Across All Variants")

    metric_cols = ["precision", "recall", "f1", "mean_iou"]
    fig = go.Figure()
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#00BCD4"]

    for i, (_, row) in enumerate(comp_df.iterrows()):
        fig.add_trace(go.Bar(
            name=display_name(row["variant"]),
            x=["Precision", "Recall", "F1 Score", "Mean IoU"],
            y=[float(row[c]) for c in metric_cols],
            text=[f"{float(row[c]):.3f}" for c in metric_cols],
            textposition="outside",
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score (0.0 – 1.0)", range=[0, 1.12]),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Detailed table ──
    st.subheader("Detailed Metrics Table")
    display_df = comp_df.copy()
    display_df["variant"] = display_df["variant"].apply(display_name)
    for c in metric_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{float(x):.4f}")
    for c in ["tp", "fp", "fn", "total_gt"]:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda x: f"{int(x):,}")

    display_df = display_df.rename(columns={
        "variant": "Detector Variant",
        "total_gt": "Total GT Faces",
        "tp": "True Positives",
        "fp": "False Positives",
        "fn": "False Negatives",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "mean_iou": "Mean IoU",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── TP/FP/FN stacked bar ──
    st.subheader("Detection Outcome Breakdown")
    st.markdown("""
    <div class='info-box'>
    <b>TP (True Positive):</b> Model correctly found a real face (IoU > 0.5) &nbsp;|&nbsp;
    <b>FP (False Positive):</b> Model drew a box where there's no face &nbsp;|&nbsp;
    <b>FN (False Negative):</b> A real face the model missed entirely
    </div>
    """, unsafe_allow_html=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="True Positives (TP)",
        x=[display_name(r["variant"]) for _, r in comp_df.iterrows()],
        y=comp_df["tp"].astype(int),
        marker_color="#4CAF50",
        text=comp_df["tp"].astype(int).apply(lambda x: f"{x:,}"),
        textposition="inside",
    ))
    fig2.add_trace(go.Bar(
        name="False Negatives (FN — missed)",
        x=[display_name(r["variant"]) for _, r in comp_df.iterrows()],
        y=comp_df["fn"].astype(int),
        marker_color="#F44336",
        text=comp_df["fn"].astype(int).apply(lambda x: f"{x:,}"),
        textposition="inside",
    ))
    fig2.add_trace(go.Bar(
        name="False Positives (FP — wrong)",
        x=[display_name(r["variant"]) for _, r in comp_df.iterrows()],
        y=comp_df["fp"].astype(int),
        marker_color="#9C27B0",
        text=comp_df["fp"].astype(int).apply(lambda x: f"{x:,}"),
        textposition="inside",
    ))
    fig2.update_layout(
        barmode="stack",
        yaxis=dict(title="Number of Faces"),
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)



    # ── Sub-analyses in tabs ──
    st.markdown("---")
    _show_tab = st.radio(
        "Explore deeper →",
        ["📈 Crowd Density Impact", "📉 PR Curve & Threshold", "🏷️ Attribute Recall"],
        horizontal=True,
        key="showdown_subtab",
    )

    if _show_tab == "📈 Crowd Density Impact":
        st.subheader("📈 Recall by Face Density")
        st.markdown("""
        <div class="info-box">
        Images grouped by how many faces they contain. <b>Recall drops sharply in crowded scenes</b>
        because faces overlap, become smaller, and NMS suppresses valid detections.
        </div>
        """, unsafe_allow_html=True)

        fig_gw = go.Figure()
        _gw_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#00BCD4"]
        for i, (v, data) in enumerate(variants.items()):
            gw = data["groupwise"]
            if gw.empty:
                continue
            fig_gw.add_trace(go.Scatter(
                x=gw["bin"], y=gw["recall"],
                mode="lines+markers",
                name=display_name(v),
                line=dict(width=3, color=_gw_colors[i % len(_gw_colors)]),
                marker=dict(size=10),
            ))
        fig_gw.update_layout(
            xaxis_title="Number of Faces per Image",
            yaxis=dict(title="Recall", range=[0, 1.05]),
            height=480, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_gw, use_container_width=True)

    elif _show_tab == "📉 PR Curve & Threshold":
        _pr_v = st.selectbox("Select variant", variant_names, format_func=display_name, key="pr_show")
        pr_data = variants[_pr_v]["pr_curve"]
        thresh_data = variants[_pr_v]["threshold"]
        _ap = float(variants[_pr_v]["metrics"].iloc[0].get("ap", 0)) if "ap" in variants[_pr_v]["metrics"].columns else 0.0

        if not pr_data.empty:
            st.subheader(f"Precision-Recall Curve (AP = {_ap:.4f})")
            fig_prc = go.Figure(go.Scatter(
                x=pr_data["recall"], y=pr_data["precision"],
                mode="lines", fill="tozeroy",
                line=dict(color="#2196F3", width=2),
                fillcolor="rgba(33,150,243,0.12)",
            ))
            fig_prc.update_layout(
                xaxis=dict(title="Recall", range=[0, 1.02]),
                yaxis=dict(title="Precision", range=[0, 1.05]),
                height=450, template="plotly_white",
            )
            st.plotly_chart(fig_prc, use_container_width=True)

        if not thresh_data.empty:
            st.subheader("Threshold Sensitivity")
            fig_th = go.Figure()
            for col, color, nm in [("precision","#2196F3","Precision"),("recall","#F44336","Recall"),("f1","#4CAF50","F1")]:
                fig_th.add_trace(go.Scatter(x=thresh_data["threshold"], y=thresh_data[col],
                    mode="lines+markers", name=nm, line=dict(color=color, width=2), marker=dict(size=7)))
            _bi = thresh_data["f1"].idxmax()
            fig_th.add_vline(x=thresh_data.loc[_bi,"threshold"], line_dash="dash", line_color="green",
                             annotation_text=f"Best F1={thresh_data.loc[_bi,'f1']:.3f}")
            fig_th.update_layout(xaxis_title="Confidence Threshold",
                yaxis=dict(title="Score", range=[0,1.1]), height=450, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_th, use_container_width=True)

    elif _show_tab == "🏷️ Attribute Recall":
        _at_v = st.selectbox("Select variant", variant_names, format_func=display_name, key="attr_show")
        _at_data = variants[_at_v]["attribute"]
        if _at_data.empty:
            st.warning("No attribute data available.")
        else:
            for attr in _at_data["attribute"].unique():
                subset = _at_data[_at_data["attribute"] == attr]
                st.subheader(f"{attr.capitalize()}")
                _attr_colors = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
                fig_at = go.Figure(go.Bar(
                    x=subset["label"], y=subset["recall"],
                    text=[f"{r:.3f} (n={t:,})" for r, t in zip(subset["recall"], subset["total"])],
                    textposition="outside",
                    marker_color=[_attr_colors.get(l, "#2196F3") for l in subset["level"]],
                ))
                fig_at.update_layout(yaxis=dict(title="Recall", range=[0,1.15]), height=320, template="plotly_white")
                st.plotly_chart(fig_at, use_container_width=True)



# ══════════════════════════════════════════════════════════════
# PAGE 2: Group-wise Analysis
# ══════════════════════════════════════════════════════════════
elif page == "_groupwise":  # merged into Detector Showdown
    st.title("📈 Group-wise Analysis — Performance vs. Face Density")

    st.markdown("""
    <div class='info-box'>
    Images are grouped by how many ground-truth faces they contain.<br>
    <b>Face density bins:</b> 0–10, 11–20, 21–30, 31–40, 41–50, 51+<br>
    <b>Expectation:</b> Recall drops as images get more crowded because faces overlap, become smaller, and NMS suppresses valid detections.
    </div>
    """, unsafe_allow_html=True)

    # ── Recall line chart ──
    st.subheader("Recall by Face Density")
    st.caption("Recall = fraction of real faces the model found. X-axis = number of GT faces per image.")

    fig = go.Figure()
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for i, (v, data) in enumerate(variants.items()):
        gw = data["groupwise"]
        if gw.empty:
            continue
        fig.add_trace(go.Scatter(
            x=gw["bin"], y=gw["recall"],
            mode="lines+markers+text",
            name=display_name(v),
            text=[f"{r:.2f}" for r in gw["recall"]],
            textposition="top center",
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=10),
        ))
    fig.update_layout(
        xaxis=dict(title="Number of Ground-Truth Faces per Image (bin)"),
        yaxis=dict(title="Recall (0.0 – 1.0)", range=[0, 1.05]),
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── F1 line chart ──
    st.subheader("F1 Score by Face Density")
    st.caption("F1 = harmonic mean of Precision and Recall. Balances both metrics.")

    fig2 = go.Figure()
    for i, (v, data) in enumerate(variants.items()):
        gw = data["groupwise"]
        if gw.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=gw["bin"], y=gw["f1"],
            mode="lines+markers+text",
            name=display_name(v),
            text=[f"{r:.2f}" for r in gw["f1"]],
            textposition="top center",
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=10, symbol="square"),
        ))
    fig2.update_layout(
        xaxis=dict(title="Number of Ground-Truth Faces per Image (bin)"),
        yaxis=dict(title="F1 Score (0.0 – 1.0)", range=[0, 1.05]),
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw data tables ──
    st.subheader("Raw Group-wise Data")
    selected_v = st.selectbox("Select variant", variant_names, format_func=display_name)
    gw = variants[selected_v]["groupwise"]
    if not gw.empty:
        display_gw = gw.rename(columns={
            "bin": "Face Count Bin",
            "num_images": "Images",
            "total_gt": "Total GT Faces",
            "tp": "TP", "fp": "FP", "fn": "FN",
            "precision": "Precision", "recall": "Recall",
            "f1": "F1", "mean_iou": "Mean IoU",
        })
        st.dataframe(display_gw, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3: Quality Analysis
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Why Faces Are Missed":
    st.title("🔬 Why Faces Are Missed — Root Cause Analysis")

    st.markdown("""
    <div class='info-box'>
    For every face (detected, undetected, or false positive), we compute quality features
    to understand <b>why</b> the detector succeeds or fails.<br><br>
    <b>Metrics measured:</b><br>
    • <b>Face Area</b> — bounding box width × height in <b>pixels²</b>.
      Example: 256 px² ≈ 16×16 face (tiny), 1024 px² ≈ 32×32 (small), 4096 px² ≈ 64×64 (medium).<br>
    • <b>Face Brightness</b> — mean pixel value of the HSV V-channel inside the face box.
      Scale: <b>0 = black, 255 = pure white</b>. Below 80 → dark face (harder to detect).<br>
    • <b>Image Brightness</b> — same as above but for the whole image.<br>
    • <b>Image Blur</b> — variance of the Laplacian operator on the grayscale image.
      <b>Higher = sharper</b>. Below 50 → blurry; above 200 → sharp.<br>
    • <b>Confidence</b> — model's certainty for a detection, from <b>0.0 (uncertain) to 1.0 (certain)</b>.
    </div>
    """, unsafe_allow_html=True)

    selected_v = st.selectbox("Select detector variant", variant_names, format_func=display_name, key="quality_variant")
    q_df = variants[selected_v]["quality"]

    if q_df.empty:
        st.warning("No quality data available for this variant.")
    else:
        # ── Summary cards ──
        st.subheader("Category Breakdown")
        cat_counts = q_df["category"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Detected (TP)", f"{cat_counts.get('detected', 0):,}")
        c2.metric("❌ Undetected (FN)", f"{cat_counts.get('undetected', 0):,}")
        c3.metric("⚠️ False Positive (FP)", f"{cat_counts.get('false_positive', 0):,}")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Face size distribution ──
        st.subheader("Face Size Distribution — Detected vs. Missed")
        st.caption("Face area = bounding box width × height in pixels². "
                   "Reference: 16×16 = 256 px², 32×32 = 1,024 px², 64×64 = 4,096 px²")

        fig = go.Figure()
        for cat, color, label in [
            ("detected", "#4CAF50", "Detected (TP)"),
            ("undetected", "#F44336", "Undetected (FN)"),
        ]:
            subset = q_df[q_df["category"] == cat]["face_area"].clip(upper=12000)
            if not subset.empty:
                fig.add_trace(go.Histogram(
                    x=subset, nbinsx=80,
                    name=f"{label} (n={len(subset):,})",
                    marker_color=color, opacity=0.55,
                ))

        # Reference lines
        for area, label in [(256, "16×16 px"), (1024, "32×32 px"), (4096, "64×64 px")]:
            fig.add_vline(x=area, line_dash="dash", line_color="gray", opacity=0.6,
                          annotation_text=label, annotation_position="top")

        fig.update_layout(
            barmode="overlay",
            xaxis=dict(title="Face Bounding-Box Area (pixels²)"),
            yaxis=dict(title="Number of Faces"),
            height=450, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Size breakdown stats ──
        missed = q_df[q_df["category"] == "undetected"]
        if not missed.empty:
            n = len(missed)
            tiny_pct = (missed["face_area"] < 256).sum() / n * 100
            small_pct = (missed["face_area"] < 1024).sum() / n * 100
            med_pct = (missed["face_area"] < 4096).sum() / n * 100
            st.info(
                f"**Of {n:,} missed faces:** "
                f"{tiny_pct:.1f}% are < 16×16 px, "
                f"{small_pct:.1f}% are < 32×32 px, "
                f"{med_pct:.1f}% are < 64×64 px. "
                f"**Face size is the dominant failure factor.**"
            )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Brightness vs Size scatter ──
        st.subheader("Face Brightness vs. Face Size")
        st.caption("Brightness = mean HSV V-channel (0=black, 255=white). "
                   "The red shaded zone marks faces < 32×32 px that are hardest to detect.")

        fig2 = make_subplots()
        for cat, color, label in [
            ("detected", "#4CAF50", "Detected (TP)"),
            ("undetected", "#F44336", "Undetected (FN)"),
            ("false_positive", "#9C27B0", "False Positive (FP)"),
        ]:
            subset = q_df[q_df["category"] == cat]
            if subset.empty:
                continue
            # Sample for performance if too large
            plot_data = subset.sample(min(3000, len(subset)), random_state=42) if len(subset) > 3000 else subset
            fig2.add_trace(go.Scattergl(
                x=plot_data["face_brightness"],
                y=plot_data["face_area"].clip(upper=15000),
                mode="markers",
                name=f"{label} (n={len(subset):,})",
                marker=dict(color=color, size=4, opacity=0.35),
            ))

        fig2.add_hrect(y0=0, y1=1024, fillcolor="red", opacity=0.04, line_width=0)
        fig2.add_annotation(x=20, y=800, text="Small-face danger zone (< 32×32 px)",
                            showarrow=False, font=dict(size=10, color="red"))

        fig2.update_layout(
            xaxis=dict(title="Face Brightness — HSV V-channel (0 = black, 255 = white)"),
            yaxis=dict(title="Face Area (pixels²) — clipped at 15,000"),
            height=500, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Box plots ──
        st.subheader("Quality Metrics by Detection Outcome")

        metric_options = {
            "Face Area (pixels²)": "face_area",
            "Face Brightness (HSV V: 0=black, 255=white)": "face_brightness",
            "Image Brightness (HSV V: 0=black, 255=white)": "image_brightness",
            "Image Sharpness (Laplacian variance: higher=sharper)": "image_blur",
        }

        sel_metric_label = st.selectbox("Select metric", list(metric_options.keys()))
        sel_metric = metric_options[sel_metric_label]

        fig3 = go.Figure()
        cat_order = ["detected", "undetected", "false_positive"]
        cat_colors = {"detected": "#4CAF50", "undetected": "#F44336", "false_positive": "#9C27B0"}
        cat_nice = {"detected": "Detected (TP)", "undetected": "Undetected (FN)", "false_positive": "False Positive (FP)"}

        for cat in cat_order:
            subset = q_df[q_df["category"] == cat][sel_metric].dropna()
            if subset.empty:
                continue
            fig3.add_trace(go.Box(
                y=subset, name=cat_nice[cat],
                marker_color=cat_colors[cat],
                boxmean=True,
            ))

        fig3.update_layout(
            yaxis=dict(title=sel_metric_label),
            height=450, template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Confidence distribution ──
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.subheader("Confidence Score Distribution (TP vs FP)")
        st.caption("Confidence = model's certainty (0.0 – 1.0). "
                   "If FP confidence is high, the threshold may need raising.")

        fig4 = go.Figure()
        for cat, color, label in [
            ("detected", "#4CAF50", "True Positives"),
            ("false_positive", "#9C27B0", "False Positives"),
        ]:
            conf = q_df[q_df["category"] == cat]["confidence"]
            conf = conf[conf >= 0]
            if conf.empty:
                continue
            fig4.add_trace(go.Histogram(
                x=conf, nbinsx=50,
                name=f"{label} (n={len(conf):,}, mean={conf.mean():.3f})",
                marker_color=color, opacity=0.6,
            ))

        fig4.update_layout(
            barmode="overlay",
            xaxis=dict(title="Detection Confidence Score (0.0 – 1.0)"),
            yaxis=dict(title="Number of Detections"),
            height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ── Quality summary table ──
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.subheader("Quality Summary Table")
        qs = variants[selected_v]["quality_summary"]
        if not qs.empty:
            qs_display = qs.rename(columns={
                "category": "Category",
                "count": "Count",
                "mean_face_area": "Mean Face Area (px²)",
                "mean_face_brightness": "Mean Face Brightness (V: 0–255)",
                "mean_image_brightness": "Mean Image Brightness (V: 0–255)",
                "mean_image_blur": "Mean Image Sharpness (Laplacian var)",
                "mean_confidence": "Mean Confidence (0–1)",
            })
            st.dataframe(qs_display, use_container_width=True, hide_index=True)



    # ── Additional analysis tabs ──
    st.markdown("---")
    _fail_tab = st.radio(
        "More analysis →",
        ["📏 Proximity & Occlusion", "🎭 Event Categories"],
        horizontal=True,
        key="failure_subtab",
    )

    if _fail_tab == "📏 Proximity & Occlusion":
        st.subheader("📏 Distance-Based Failure Analysis")
        st.markdown("""
        <div class="info-box">
        <b>Euclidean distance</b> between face-pair centres reveals <b>NMS suppression</b> patterns:
        when a missed face is very close to a detected one, it was likely suppressed by Non-Maximum Suppression
        or hidden by occlusion.
        </div>
        """, unsafe_allow_html=True)

        _dv = st.selectbox("Select variant", variant_names, format_func=display_name, key="dist_fail")
        dist_df = variants[_dv]["distance"]
        if not dist_df.empty:
            _pair_colors = {"detected-detected":"#4CAF50","detected-undetected":"#FF9800","undetected-undetected":"#F44336"}
            _pair_labels = {"detected-detected":"Both Found","detected-undetected":"One Missed (NMS risk)","undetected-undetected":"Both Missed (dense cluster)"}
            fig_dist = go.Figure()
            for pt in sorted(dist_df["pair_type"].unique()):
                s = dist_df[dist_df["pair_type"]==pt]["distance"]
                fig_dist.add_trace(go.Histogram(x=s, nbinsx=60,
                    name=f"{_pair_labels.get(pt,pt)} (n={len(s):,})",
                    marker_color=_pair_colors.get(pt,"gray"), opacity=0.55))
            fig_dist.update_layout(barmode="overlay",
                xaxis_title="Distance Between Face Centres (px)",
                yaxis_title="Pairs", height=450, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_dist, use_container_width=True)

            det_undet = dist_df[dist_df["pair_type"]=="detected-undetected"]["distance"]
            if not det_undet.empty:
                close = (det_undet < 50).sum()
                st.markdown(f"""
                <div class="callout-warn">
                <b>{close:,}</b> missed faces are within 50 px of a detected face — likely
                <b>NMS suppression</b> or <b>mutual occlusion</b>.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No distance data available for this variant.")

    elif _fail_tab == "🎭 Event Categories":
        st.subheader("🎭 Per-Event Recall")
        _ev = st.selectbox("Select variant", variant_names, format_func=display_name, key="ev_fail")
        ev_df = variants[_ev]["event"]
        if not ev_df.empty:
            ev_sorted = ev_df.sort_values("recall", ascending=True)
            fig_ev = go.Figure(go.Bar(
                y=ev_sorted["event"], x=ev_sorted["recall"], orientation="h",
                text=[f"{r:.3f} ({n} imgs)" for r,n in zip(ev_sorted["recall"], ev_sorted["num_images"])],
                textposition="outside",
                marker=dict(color=ev_sorted["recall"], colorscale="RdYlGn", showscale=True,
                            colorbar=dict(title="Recall")),
            ))
            fig_ev.update_layout(xaxis=dict(title="Recall", range=[0,1.15]),
                height=max(400, len(ev_sorted)*22), template="plotly_white", margin=dict(l=180))
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info("No event data available.")



# ══════════════════════════════════════════════════════════════
# PAGE 4: Distance Analysis
# ══════════════════════════════════════════════════════════════
elif page == "_distance":  # merged
    st.title("📏 Distance-Based Failure Analysis")

    st.markdown("""
    <div class='info-box'>
    For each image, we compute <b>Euclidean distance</b> (in pixels) between the centres of
    every pair of face bounding boxes, then label each pair:<br><br>
    • <b>Detected ↔ Detected:</b> Both faces found — baseline spacing.<br>
    • <b>Detected ↔ Undetected:</b> One found, one missed. <b>Short distance here suggests
      NMS suppression or occlusion</b> — the missed face was right next to a detected one.<br>
    • <b>Undetected ↔ Undetected:</b> Both missed. Clustering here means a <b>dense crowd region</b>
      the detector cannot resolve.
    </div>
    """, unsafe_allow_html=True)

    selected_v = st.selectbox("Select detector variant", variant_names, format_func=display_name, key="dist_variant")
    dist_df = variants[selected_v]["distance"]
    dist_sum = variants[selected_v]["distance_summary"]

    if dist_df.empty:
        st.warning("No distance data available for this variant.")
    else:
        # ── Summary table ──
        st.subheader("Distance Summary by Pair Type")
        if not dist_sum.empty:
            ds_display = dist_sum.rename(columns={
                "pair_type": "Pair Type",
                "count": "Number of Pairs",
                "mean_distance": "Mean Distance (px)",
                "median_distance": "Median Distance (px)",
                "std_distance": "Std Dev (px)",
            })
            st.dataframe(ds_display, use_container_width=True, hide_index=True)

        # ── Histogram ──
        st.subheader("Distance Distribution by Pair Type")
        st.caption("Distance = Euclidean distance between face bounding-box centres, in pixels. "
                   "Short Det↔Undet distance → likely NMS or occlusion failure.")

        pair_colors = {
            "detected-detected": "#4CAF50",
            "detected-undetected": "#FF9800",
            "undetected-undetected": "#F44336",
        }
        pair_labels = {
            "detected-detected": "Detected ↔ Detected",
            "detected-undetected": "Detected ↔ Undetected (NMS/occlusion risk)",
            "undetected-undetected": "Undetected ↔ Undetected (dense cluster)",
        }

        fig = go.Figure()
        for pt in sorted(dist_df["pair_type"].unique()):
            subset = dist_df[dist_df["pair_type"] == pt]["distance"]
            fig.add_trace(go.Histogram(
                x=subset, nbinsx=60,
                name=f"{pair_labels.get(pt, pt)} (n={len(subset):,})",
                marker_color=pair_colors.get(pt, "gray"),
                opacity=0.55,
            ))

        fig.update_layout(
            barmode="overlay",
            xaxis=dict(title="Euclidean Distance Between Face Centres (pixels)"),
            yaxis=dict(title="Number of Face Pairs"),
            height=500, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Key insight ──
        det_undet = dist_df[dist_df["pair_type"] == "detected-undetected"]["distance"]
        if not det_undet.empty:
            close = (det_undet < 50).sum()
            st.info(
                f"**{close:,}** detected↔undetected pairs are within 50 pixels — "
                f"these misses are likely caused by NMS suppression or mutual occlusion."
            )


# ══════════════════════════════════════════════════════════════
# PAGE 5: Attribute Analysis
# ══════════════════════════════════════════════════════════════
elif page == "_attribute":  # merged
    st.title("🏷️ Attribute-Based Detection Analysis")

    st.markdown("""
    <div class='info-box'>
    WIDER FACE provides per-face annotation attributes. We compute <b>recall</b>
    (fraction of real faces found) broken down by each attribute level:<br><br>
    • <b>Blur:</b> 0 = Clear, 1 = Normal Blur, 2 = Heavy Blur<br>
    • <b>Occlusion:</b> 0 = None, 1 = Partial, 2 = Heavy<br>
    • <b>Illumination:</b> 0 = Normal, 1 = Extreme<br>
    • <b>Expression:</b> 0 = Typical, 1 = Exaggerated<br>
    • <b>Pose:</b> 0 = Typical, 1 = Atypical<br><br>
    This reveals which conditions hurt detection the most.
    </div>
    """, unsafe_allow_html=True)

    selected_v = st.selectbox("Select detector variant", variant_names, format_func=display_name, key="attr_variant")
    attr_df = variants[selected_v]["attribute"]

    if attr_df.empty:
        st.warning("No attribute data available. Re-run the pipeline to generate it.")
    else:
        # One chart per attribute
        for attr in attr_df["attribute"].unique():
            subset = attr_df[attr_df["attribute"] == attr]

            st.subheader(f"{attr.capitalize()} — Recall by Level")

            fig = go.Figure()
            colors_map = {
                0: "#4CAF50",  # good (clear, none, normal, typical)
                1: "#FF9800",  # moderate
                2: "#F44336",  # hard
            }
            fig.add_trace(go.Bar(
                x=subset["label"],
                y=subset["recall"],
                text=[f"{r:.3f}<br>(n={t:,})" for r, t in zip(subset["recall"], subset["total"])],
                textposition="outside",
                marker_color=[colors_map.get(l, "#2196F3") for l in subset["level"]],
            ))
            fig.update_layout(
                yaxis=dict(title="Recall (0.0 – 1.0)", range=[0, 1.15]),
                xaxis=dict(title=f"{attr.capitalize()} Level"),
                height=350, template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary insight
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.subheader("Key Takeaways")
        for attr in attr_df["attribute"].unique():
            subset = attr_df[attr_df["attribute"] == attr]
            best = subset.loc[subset["recall"].idxmax()]
            worst = subset.loc[subset["recall"].idxmin()]
            if best["label"] != worst["label"]:
                gap = best["recall"] - worst["recall"]
                st.markdown(
                    f"- **{attr.capitalize()}**: Best = {best['label']} "
                    f"(recall {best['recall']:.3f}), Worst = {worst['label']} "
                    f"(recall {worst['recall']:.3f}), gap = **{gap:.3f}**"
                )


# ══════════════════════════════════════════════════════════════
# PAGE 6: Event/Scene Analysis
# ══════════════════════════════════════════════════════════════
elif page == "🌆 Scene & Crowd Analysis":
    st.title("🌆 Scene & Crowd Analysis")
    st.markdown("""
    <div class="info-box">
    WIDER FACE images span <b>61 real-world event categories</b> (Parade, Meeting, Basketball, Festival…).
    This page reveals which scenes are hardest for detection, and how <b>crowd density</b> degrades recall.
    Understanding scene difficulty helps prioritize where SR-based improvements matter most.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    WIDER FACE images are organized into <b>61 event categories</b> (e.g., Parade, Meeting,
    Dresses, Festival, etc.). We compute <b>precision, recall, and F1</b> per category to
    reveal which scene types are easiest/hardest for the detector.
    </div>
    """, unsafe_allow_html=True)

    selected_v = st.selectbox("Select detector variant", variant_names, format_func=display_name, key="event_variant")
    ev_df = variants[selected_v]["event"]

    if ev_df.empty:
        st.warning("No event data available. Re-run the pipeline to generate it.")
    else:
        # Sort by recall for display
        ev_sorted = ev_df.sort_values("recall", ascending=True)

        st.subheader("Recall by Event Category (sorted worst → best)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ev_sorted["event"],
            x=ev_sorted["recall"],
            orientation="h",
            text=[f"{r:.3f} ({n} imgs)" for r, n in zip(ev_sorted["recall"], ev_sorted["num_images"])],
            textposition="outside",
            marker=dict(
                color=ev_sorted["recall"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Recall"),
            ),
        ))
        fig.update_layout(
            xaxis=dict(title="Recall (0.0 – 1.0)", range=[0, 1.15]),
            yaxis=dict(title=""),
            height=max(400, len(ev_sorted) * 22),
            template="plotly_white",
            margin=dict(l=180),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top/bottom 5
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("5 Hardest Categories")
            hard = ev_sorted.head(5)[["event", "recall", "num_images", "total_gt"]].rename(columns={
                "event": "Event", "recall": "Recall", "num_images": "Images", "total_gt": "GT Faces",
            })
            st.dataframe(hard, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("5 Easiest Categories")
            easy = ev_sorted.tail(5)[["event", "recall", "num_images", "total_gt"]].rename(columns={
                "event": "Event", "recall": "Recall", "num_images": "Images", "total_gt": "GT Faces",
            })
            st.dataframe(easy, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 7: PR Curve & Threshold Sensitivity
# ══════════════════════════════════════════════════════════════
elif page == "_pr_curve":  # merged
    st.title("📉 Precision-Recall Curve & Threshold Sensitivity")

    st.markdown("""
    <div class='info-box'>
    <b>PR Curve:</b> Shows how precision trades off against recall as the detection confidence
    threshold varies. The area under this curve is the <b>Average Precision (AP)</b> — a single
    number summarizing overall detector quality.<br><br>
    <b>Threshold Sensitivity:</b> How precision, recall, and F1 change as we raise/lower the
    confidence cutoff. Helps choose the optimal operating point.
    </div>
    """, unsafe_allow_html=True)

    selected_v = st.selectbox("Select detector variant", variant_names, format_func=display_name, key="pr_variant")
    pr_df = variants[selected_v]["pr_curve"]
    thresh_df = variants[selected_v]["threshold"]
    metrics = variants[selected_v]["metrics"]
    ap_val = float(metrics.iloc[0].get("ap", 0)) if "ap" in metrics.columns else 0.0

    # PR Curve
    if not pr_df.empty:
        st.subheader(f"Precision-Recall Curve (AP = {ap_val:.4f})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pr_df["recall"], y=pr_df["precision"],
            mode="lines", fill="tozeroy",
            name=f"AP = {ap_val:.4f}",
            line=dict(color="#2196F3", width=2),
            fillcolor="rgba(33, 150, 243, 0.15)",
        ))
        fig.update_layout(
            xaxis=dict(title="Recall (fraction of real faces found)", range=[0, 1.02]),
            yaxis=dict(title="Precision (fraction of correct detections)", range=[0, 1.05]),
            height=500, template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Average Precision (AP)", f"{ap_val:.4f}")
    else:
        st.warning("No PR curve data available.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Threshold sensitivity
    if not thresh_df.empty:
        st.subheader("Confidence Threshold Sensitivity")
        st.caption("How metrics change as we vary the minimum confidence required for a detection.")

        fig2 = go.Figure()
        for col, color, name in [
            ("precision", "#2196F3", "Precision"),
            ("recall", "#F44336", "Recall"),
            ("f1", "#4CAF50", "F1 Score"),
        ]:
            fig2.add_trace(go.Scatter(
                x=thresh_df["threshold"], y=thresh_df[col],
                mode="lines+markers+text",
                name=name,
                text=[f"{v:.3f}" for v in thresh_df[col]],
                textposition="top center",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

        # Mark optimal F1
        best_idx = thresh_df["f1"].idxmax()
        best_t = thresh_df.loc[best_idx, "threshold"]
        best_f1 = thresh_df.loc[best_idx, "f1"]
        fig2.add_vline(x=best_t, line_dash="dash", line_color="green", opacity=0.6,
                       annotation_text=f"Best F1={best_f1:.3f} @ thresh={best_t}")

        fig2.update_layout(
            xaxis=dict(title="Confidence Threshold"),
            yaxis=dict(title="Score (0.0 – 1.0)", range=[0, 1.1]),
            height=500, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(thresh_df.rename(columns={
            "threshold": "Threshold", "tp": "TP", "fp": "FP", "fn": "FN",
            "precision": "Precision", "recall": "Recall", "f1": "F1",
        }), use_container_width=True, hide_index=True)
    else:
        st.warning("No threshold sensitivity data available.")


# ══════════════════════════════════════════════════════════════
# PAGE 8: Improvement Experiments
# ══════════════════════════════════════════════════════════════
elif page == "🧬 AdaSR Ablation Study":
    st.title("🧬 AdaSR-Face — Ablation Study")
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);">
        <h1 style="font-size:22px;">Ablation Study: Does Super-Resolution Help Detect Small Faces?</h1>
        <p>
            94% of missed faces are &lt; 32×32 px. We test whether <b>learned super-resolution</b>
            (Real-ESRGAN) can recover them — comparing blind full-image SR, bicubic upscaling,
            and our novel <b>AdaSR-Face</b> pipeline that selectively upscales only the regions
            where the detector is least confident.
        </p>
        <span class="hero-tag">SCRFD · Real-ESRGAN · Tiled Detection · Soft-NMS Fusion</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Load ablation data from individual CSVs ──
    import glob as _glob
    _abl_files = sorted(_glob.glob(str(EXP_CSV_DIR / "ablation_E*.csv")))
    _exp_files_old = EXP_CSV_DIR / "experiment_summary.csv"

    abl_df = None
    if _abl_files:
        _parts = []
        for _f in _abl_files:
            _tmp = pd.read_csv(_f)
            if "experiment" in _tmp.columns and _tmp["experiment"].notna().any():
                _parts.append(_tmp)
        if _parts:
            abl_df = pd.concat(_parts, ignore_index=True).drop_duplicates(subset="experiment")

    # Fallback: old experiment_summary.csv
    if abl_df is None and _exp_files_old.exists():
        abl_df = pd.read_csv(_exp_files_old)

    if abl_df is not None and len(abl_df) > 0:
        # Add short labels
        _label_map = {
            "E1_scrfd_baseline": "E1 Baseline",
            "E2_scrfd_tiled_ms": "E2 Tiled+MS",
            "E3_scrfd_bicubic": "E3 Bicubic SR",
            "E4_scrfd_blind_sr": "E4 Blind SR",
            "E5_scrfd_adaptive_sr": "E5 AdaSR ★",
            "E6_scrfd_adasr_tiled_ms": "E6 AdaSR+Tiled",
            "E7_det10g_adasr_tiled_ms": "E7 det10g+AdaSR",
        }
        abl_df["label"] = abl_df["experiment"].map(_label_map).fillna(abl_df["experiment"])

        best_row = abl_df.loc[abl_df["f1"].idxmax()]
        baseline_row = abl_df[abl_df["experiment"].str.contains("E1", na=False)]
        baseline_f1 = baseline_row["f1"].values[0] if len(baseline_row) > 0 else 0.682
        baseline_recall = baseline_row["recall"].values[0] if len(baseline_row) > 0 else 0.521

        # ── Hero KPI row ──
        st.subheader("🏆 Best Result")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Best Config", best_row["label"])
        k2.metric("F1 Score", f"{best_row['f1']:.4f}",
                  delta=f"{best_row['f1'] - baseline_f1:+.4f} vs baseline")
        k3.metric("Recall", f"{best_row['recall']:.4f}",
                  delta=f"{best_row['recall'] - baseline_recall:+.4f}")
        k4.metric("Precision", f"{best_row['precision']:.4f}")
        if "ap" in best_row.index and pd.notna(best_row.get("ap")):
            k5.metric("AP", f"{best_row['ap']:.4f}")
        else:
            k5.metric("Experiments", f"{len(abl_df)}")

        # Previous best reference
        st.markdown(
            '<div class="info-box">📌 <b>Previous best:</b> RetinaFace Tiled+MS — '
            'F1 = 0.802, Recall = 0.711 (full WIDER FACE val set)</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Comparison Table ──
        st.subheader("📋 Ablation Comparison Table")
        _tbl_cols = ["label", "sr_mode", "tiled", "multiscale",
                     "precision", "recall", "f1", "ap", "tp", "fp", "fn", "elapsed_s"]
        _tbl_avail = [c for c in _tbl_cols if c in abl_df.columns]
        _tbl = abl_df[_tbl_avail].copy()
        _rename = {"label": "Experiment", "sr_mode": "SR", "tiled": "Tiled",
                   "multiscale": "MS", "precision": "P", "recall": "R",
                   "elapsed_s": "Time (s)"}
        _tbl = _tbl.rename(columns={k: v for k, v in _rename.items() if k in _tbl.columns})

        def _hl_best(s):
            if s.name in ["f1", "R", "ap"]:
                best = s.max()
                return ["background-color: #2ecc71; color: white; font-weight: bold"
                        if v == best else "" for v in s]
            return [""] * len(s)

        st.dataframe(_tbl.style.apply(_hl_best).format(
            {c: "{:.4f}" for c in ["P", "R", "f1", "ap"] if c in _tbl.columns}
        ), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── F1 Bar Chart ──
        st.subheader("📊 F1 Score by Experiment")
        _colors_f1 = ["#2ecc71" if e == best_row["experiment"] else "#3498db"
                      for e in abl_df["experiment"]]
        fig_f1 = go.Figure()
        fig_f1.add_trace(go.Bar(
            x=abl_df["label"], y=abl_df["f1"],
            marker_color=_colors_f1,
            text=abl_df["f1"].round(4), textposition="outside",
        ))
        fig_f1.add_hline(y=0.802, line_dash="dash", line_color="red",
                         annotation_text="Previous Best F1=0.802")
        fig_f1.add_hline(y=baseline_f1, line_dash="dot", line_color="orange",
                         annotation_text=f"SCRFD Baseline F1={baseline_f1:.3f}")
        fig_f1.update_layout(
            yaxis_title="F1 Score",
            yaxis_range=[0, min(1.0, abl_df["f1"].max() * 1.15)],
            template="plotly_white", height=480,
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("---")

        # ── Size-Bucket Recall Heatmap (key for SR paper) ──
        st.subheader("🔍 Recall by Face Size — Where SR Helps")
        st.markdown("""
        This is the **core evidence** for the AdaSR paper: SR should improve recall
        specifically on small faces (0–16 px). Compare how each method performs
        across size buckets.
        """)

        _size_frames = []
        for _, _r in abl_df.iterrows():
            _sp = EXP_CSV_DIR / f"ablation_size_{_r['experiment']}.csv"
            if _sp.exists():
                _sd = pd.read_csv(_sp)
                _sd["experiment"] = _r["label"]
                _size_frames.append(_sd)

        if _size_frames:
            size_all = pd.concat(_size_frames, ignore_index=True)
            # Pivot: experiments as rows, size buckets as columns
            _pivot = size_all.pivot_table(
                index="experiment", columns="size_bucket",
                values="recall", aggfunc="first",
            )
            # Reorder columns by size
            _bucket_order = ["0-10px", "10-16px", "16-32px", "32-64px", "64-128px", "128+px"]
            _pivot = _pivot[[c for c in _bucket_order if c in _pivot.columns]]

            fig_heat = go.Figure(go.Heatmap(
                z=_pivot.values,
                x=_pivot.columns.tolist(),
                y=_pivot.index.tolist(),
                text=np.round(_pivot.values, 3),
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmin=0, zmax=1,
            ))
            fig_heat.update_layout(
                xaxis_title="Face Size Bucket",
                yaxis_title="Experiment",
                template="plotly_white", height=50 + 55 * len(_pivot),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Grouped bar chart for 0-10px and 10-16px buckets (small faces)
            st.markdown("**Small Face Recall (< 16 px) — SR Impact Zone:**")
            _small = size_all[size_all["size_bucket"].isin(["0-10px", "10-16px"])]
            if len(_small) > 0:
                fig_small = px.bar(
                    _small, x="experiment", y="recall", color="size_bucket",
                    barmode="group", text="recall",
                    color_discrete_map={"0-10px": "#e74c3c", "10-16px": "#f39c12"},
                )
                fig_small.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_small.update_layout(
                    yaxis_title="Recall", yaxis_range=[0, 1.1],
                    template="plotly_white", height=420,
                    legend_title="Size Bucket",
                )
                st.plotly_chart(fig_small, use_container_width=True)
        else:
            st.info("No per-size recall data available yet.")

        st.markdown("---")

        # ── Precision vs Recall Scatter with F1 Iso-Curves ──
        st.subheader("🎯 Precision vs Recall Trade-off")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=abl_df["recall"], y=abl_df["precision"],
            mode="markers+text",
            text=abl_df["label"], textposition="top center",
            marker=dict(size=14, color=abl_df["f1"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="F1")),
            name="Experiments",
        ))
        for _fv in [0.65, 0.70, 0.75, 0.80, 0.85]:
            _rv = np.linspace(0.01, 1.0, 200)
            _pv = (_fv * _rv) / (2 * _rv - _fv)
            _m = (_pv > 0) & (_pv <= 1)
            fig_pr.add_trace(go.Scatter(
                x=_rv[_m], y=_pv[_m], mode="lines",
                line=dict(dash="dot", width=1, color="gray"),
                name=f"F1={_fv}", showlegend=True,
            ))
        fig_pr.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            xaxis_range=[0.3, 1], yaxis_range=[0.5, 1],
            template="plotly_white", height=550,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")

        # ── Delta Analysis ──
        st.subheader("📈 Improvement Delta vs SCRFD Baseline")
        _delta = abl_df[~abl_df["experiment"].str.contains("E1_scrfd_baseline", na=False)].copy()
        if len(_delta) > 0:
            _delta["f1_delta"] = _delta["f1"] - baseline_f1
            _delta["recall_delta"] = _delta["recall"] - baseline_recall

            fig_delta = make_subplots(rows=1, cols=2,
                                      subplot_titles=["F1 Delta", "Recall Delta"])
            for ci, (met, lab) in enumerate([("f1_delta", "F1"), ("recall_delta", "Recall")], 1):
                _dc = ["#2ecc71" if v >= 0 else "#e74c3c" for v in _delta[met]]
                fig_delta.add_trace(go.Bar(
                    x=_delta["label"], y=_delta[met],
                    marker_color=_dc,
                    text=_delta[met].round(4), textposition="outside",
                    showlegend=False,
                ), row=1, col=ci)
            fig_delta.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_delta, use_container_width=True)

        st.markdown("---")

        # ── Speed vs Accuracy ──
        if "elapsed_s" in abl_df.columns and abl_df["elapsed_s"].notna().any():
            st.subheader("⏱️ Speed vs Accuracy")
            fig_spd = go.Figure()
            fig_spd.add_trace(go.Scatter(
                x=abl_df["elapsed_s"], y=abl_df["f1"],
                mode="markers+text",
                text=abl_df["label"], textposition="top center",
                marker=dict(size=16, color=abl_df["recall"], colorscale="RdYlGn",
                            showscale=True, colorbar=dict(title="Recall")),
            ))
            fig_spd.update_layout(
                xaxis_title="Runtime (seconds)", yaxis_title="F1 Score",
                template="plotly_white", height=480,
            )
            st.plotly_chart(fig_spd, use_container_width=True)
            st.markdown("---")

        # ── Per-Experiment Detail Expanders ──
        st.subheader("🔬 Per-Experiment Drill-Down")
        for _, _row in abl_df.iterrows():
            _ename = _row["experiment"]
            _elabel = _row["label"]
            _sr_tag = f"  |  SR: {_row['sr_mode']}" if "sr_mode" in _row.index and pd.notna(_row.get("sr_mode")) else ""
            with st.expander(
                f"{_elabel}: F1={_row['f1']:.4f}  |  R={_row['recall']:.4f}{_sr_tag}",
                expanded=False,
            ):
                # Config summary
                _cfg_items = {}
                for _ck in ["detector", "sr_mode", "sr_scale", "tiled", "multiscale"]:
                    if _ck in _row.index and pd.notna(_row.get(_ck)):
                        _cfg_items[_ck] = str(_row[_ck])
                if _cfg_items:
                    st.markdown("**Config:** " + " · ".join(f"`{k}={v}`" for k, v in _cfg_items.items()))

                # Description
                if "description" in _row.index and pd.notna(_row.get("description")):
                    st.caption(_row["description"])

                _ec1, _ec2 = st.columns(2)

                # Size analysis
                _sz_path = EXP_CSV_DIR / f"ablation_size_{_ename}.csv"
                if _sz_path.exists():
                    _sz = pd.read_csv(_sz_path)
                    with _ec1:
                        st.markdown("**Recall by Face Size:**")
                        fig_sz = go.Figure(go.Bar(
                            x=_sz["size_bucket"], y=_sz["recall"],
                            marker_color=["#e74c3c", "#f39c12", "#f1c40f",
                                          "#2ecc71", "#27ae60", "#16a085"][:len(_sz)],
                            text=_sz["recall"].round(3), textposition="outside",
                        ))
                        fig_sz.update_layout(
                            yaxis_range=[0, 1.15], template="plotly_white", height=320,
                            margin=dict(t=20),
                        )
                        st.plotly_chart(fig_sz, use_container_width=True, key=f"sz_{_ename}")

                # Attribute analysis
                _attr_path = EXP_CSV_DIR / f"ablation_attr_{_ename}.csv"
                if _attr_path.exists():
                    _at = pd.read_csv(_attr_path)
                    with _ec2:
                        st.markdown("**Recall by Attribute:**")
                        if "attribute" in _at.columns and "label" in _at.columns:
                            _at["attr_label"] = _at["attribute"] + ": " + _at["label"]
                            fig_at = go.Figure(go.Bar(
                                x=_at["attr_label"], y=_at["recall"],
                                marker_color="#3498db",
                                text=_at["recall"].round(3), textposition="outside",
                            ))
                            fig_at.update_layout(
                                yaxis_range=[0, 1.15], template="plotly_white", height=320,
                                margin=dict(t=20), xaxis_tickangle=-45,
                            )
                            st.plotly_chart(fig_at, use_container_width=True, key=f"at_{_ename}")

                # PR curve
                for _prefix in ["ablation_pr_", "exp_pr_curve_"]:
                    _pr_path = EXP_CSV_DIR / f"{_prefix}{_ename}.csv"
                    if _pr_path.exists():
                        _prd = pd.read_csv(_pr_path)
                        if "recall" in _prd.columns and "precision" in _prd.columns:
                            _ap_str = f" (AP={_row['ap']:.4f})" if "ap" in _row.index and pd.notna(_row.get("ap")) else ""
                            st.markdown(f"**PR Curve{_ap_str}:**")
                            fig_prc = go.Figure(go.Scatter(
                                x=_prd["recall"], y=_prd["precision"],
                                mode="lines", fill="tozeroy",
                                line=dict(color="#e74c3c"),
                            ))
                            fig_prc.update_layout(
                                xaxis_title="Recall", yaxis_title="Precision",
                                xaxis_range=[0, 1], yaxis_range=[0, 1],
                                template="plotly_white", height=320,
                            )
                            st.plotly_chart(fig_prc, use_container_width=True, key=f"pr_{_ename}")
                        break

        # ── Key Takeaways ──
        st.markdown("---")
        st.subheader("💡 Key Findings")
        if len(abl_df) > 1:
            _best = abl_df.loc[abl_df["f1"].idxmax()]
            _worst = abl_df.loc[abl_df["f1"].idxmin()]
            _hi_r = abl_df.loc[abl_df["recall"].idxmax()]
            _hi_p = abl_df.loc[abl_df["precision"].idxmax()]

            _findings = [
                f"**Best F1**: `{_best['label']}` achieved F1 = **{_best['f1']:.4f}** "
                f"({_best['f1'] - baseline_f1:+.4f} vs SCRFD baseline)",
                f"**Highest Recall**: `{_hi_r['label']}` with R = **{_hi_r['recall']:.4f}**",
                f"**Highest Precision**: `{_hi_p['label']}` with P = **{_hi_p['precision']:.4f}**",
            ]

            # SR-specific insights
            _sr_exps = abl_df[abl_df["sr_mode"].isin(["blind", "adaptive", "bicubic"])] if "sr_mode" in abl_df.columns else pd.DataFrame()
            if len(_sr_exps) > 0:
                _sr_best = _sr_exps.loc[_sr_exps["recall"].idxmax()]
                _findings.append(
                    f"**SR Impact**: Best SR method (`{_sr_best['label']}`) achieves "
                    f"R = {_sr_best['recall']:.4f} vs {baseline_recall:.4f} baseline "
                    f"(**+{_sr_best['recall'] - baseline_recall:.1%}** recall gain)"
                )

            _blind = abl_df[abl_df["sr_mode"] == "blind"] if "sr_mode" in abl_df.columns else pd.DataFrame()
            if len(_blind) > 0:
                _findings.append(
                    f"**Blind SR hurts**: Full-image SR (`{_blind.iloc[0]['label']}`) "
                    f"F1 = {_blind.iloc[0]['f1']:.4f} — upscaled images get resized "
                    f"back down by fixed detector input, losing the benefit"
                )

            for _fi in _findings:
                st.markdown(f"- {_fi}")

    else:
        st.info("No experiment results found yet.")
        st.markdown("""
        **How to run the ablation study:**
        ```bash
        # Quick test (10 images)
        python experiments/run_ablation.py --max-images 10

        # Full evaluation (3,226 images)
        python experiments/run_ablation.py

        # Specific experiments only
        python experiments/run_ablation.py --experiments E1_scrfd_baseline E5_scrfd_adaptive_sr
        ```
        """)


# ══════════════════════════════════════════════════════════════
# PAGE 9: Conclusion & Insights
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Conclusions & Next Steps":
    st.title("🧠 Conclusions & Next Steps")
    st.markdown("A data-driven summary synthesizing findings from all analysis dimensions across **all 6 detector variants** "
                "evaluated on the full **WIDER FACE** validation set (3,226 images, 39,123 valid faces).")

    # ── Load all metrics for computation ──
    all_metrics = []
    for v, data in variants.items():
        row = data["metrics"].iloc[0].to_dict()
        row["variant"] = v
        all_metrics.append(row)
    comp_df = pd.DataFrame(all_metrics)
    comp_df["f1"] = comp_df["f1"].astype(float)
    comp_df["precision"] = comp_df["precision"].astype(float)
    comp_df["recall"] = comp_df["recall"].astype(float)
    comp_df["ap"] = comp_df["ap"].astype(float) if "ap" in comp_df.columns else 0.0

    best_idx = comp_df["f1"].idxmax()
    best = comp_df.iloc[best_idx]
    best_name = display_name(best["variant"])

    # Separate RetinaFace and MTCNN variants
    rf_variants = comp_df[comp_df["variant"].str.contains("RetinaFace", case=False)]
    mt_variants = comp_df[comp_df["variant"].str.contains("MTCNN", case=False)]
    rf_base = rf_variants[~rf_variants["variant"].str.contains("Multi|Tiled", case=False)]
    rf_best = rf_variants.loc[rf_variants["f1"].idxmax()]
    mt_base = mt_variants[~mt_variants["variant"].str.contains("Multi|Tiled", case=False)]
    mt_best = mt_variants.loc[mt_variants["f1"].idxmax()]

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: Executive Summary
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("1. Executive Summary")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                border-radius: 16px; padding: 28px; color: white; margin: 15px 0;">
        <h3 style="margin-top:0; color: #e94560;">Overall Winner: {best_name}</h3>
        <table style="width:100%; color:white; font-size:16px; border-collapse:collapse;">
            <tr>
                <td style="padding:8px;"><b>F1 Score</b></td>
                <td style="padding:8px;"><b>Precision</b></td>
                <td style="padding:8px;"><b>Recall</b></td>
                <td style="padding:8px;"><b>Average Precision</b></td>
                <td style="padding:8px;"><b>True Positives</b></td>
            </tr>
            <tr style="font-size:28px; font-weight:bold;">
                <td style="padding:8px; color:#38ef7d;">{float(best['f1']):.3f}</td>
                <td style="padding:8px; color:#21CBF3;">{float(best['precision']):.3f}</td>
                <td style="padding:8px; color:#f2c94c;">{float(best['recall']):.3f}</td>
                <td style="padding:8px; color:#e94560;">{float(best['ap']):.4f}</td>
                <td style="padding:8px; color:#fff;">{int(best['tp']):,} / {int(best['total_gt']):,}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    **Key takeaway:** RetinaFace with Tiled + Multi-Scale inference is the clear winner,
    achieving an **F1 of {float(rf_best['f1']):.3f}** — a **+{(float(rf_best['f1']) - float(rf_base.iloc[0]['f1']))*100:.1f} percentage-point improvement**
    over the RetinaFace baseline ({float(rf_base.iloc[0]['f1']):.3f}). It detects **{int(rf_best['tp']):,}** of **{int(rf_best['total_gt']):,}** faces
    while maintaining **{float(rf_best['precision']):.1%} precision** — meaning only {int(rf_best['fp']):,} false alarms across 3,226 images.
    """)

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: RetinaFace vs MTCNN Head-to-Head
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("2. RetinaFace vs MTCNN — Head-to-Head")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### RetinaFace (ONNX / InsightFace)")
        st.metric("Best F1", f"{float(rf_best['f1']):.3f}", help="Tiled + MultiScale variant")
        st.metric("Baseline F1", f"{float(rf_base.iloc[0]['f1']):.3f}")
        st.metric("Best Precision", f"{float(rf_best['precision']):.3f}")
        st.metric("Best Recall", f"{float(rf_best['recall']):.3f}")
        if "ap" in rf_best:
            st.metric("Best AP", f"{float(rf_best['ap']):.4f}")
    with c2:
        st.markdown("#### MTCNN (PyTorch / facenet-pytorch)")
        st.metric("Best F1", f"{float(mt_best['f1']):.3f}", help="Best MTCNN variant")
        st.metric("Baseline F1", f"{float(mt_base.iloc[0]['f1']):.3f}")
        st.metric("Best Precision", f"{float(mt_best['precision']):.3f}")
        st.metric("Best Recall", f"{float(mt_best['recall']):.3f}")
        if "ap" in mt_best:
            st.metric("Best AP", f"{float(mt_best['ap']):.4f}")

    f1_gap = float(rf_best['f1']) - float(mt_best['f1'])
    prec_gap = float(rf_best['precision']) - float(mt_best['precision'])

    st.info(f"""
    **RetinaFace outperforms MTCNN by +{f1_gap*100:.1f}pp F1 and +{prec_gap*100:.1f}pp precision** in their best configurations.
    RetinaFace's single-stage anchor-based architecture with FPN handles multi-scale faces more effectively
    than MTCNN's three-stage cascade (P-Net → R-Net → O-Net), which struggles with very small faces and
    generates significantly more false positives when enhanced with tiling.
    """)

    # Comparison chart
    fig_compare = go.Figure()
    for i, (_, row) in enumerate(comp_df.iterrows()):
        is_rf = "RetinaFace" in row["variant"]
        fig_compare.add_trace(go.Scattergl(
            x=[float(row["recall"])],
            y=[float(row["precision"])],
            mode="markers+text",
            name=display_name(row["variant"]),
            text=[display_name(row["variant"])],
            textposition="top center" if is_rf else "bottom center",
            marker=dict(
                size=float(row["f1"]) * 40 + 5,
                color="#2196F3" if is_rf else "#FF9800",
                symbol="circle",
                line=dict(width=2, color="white"),
            ),
        ))

    fig_compare.update_layout(
        xaxis=dict(title="Recall", range=[0.25, 0.85]),
        yaxis=dict(title="Precision", range=[0.4, 1.02]),
        height=500, template="plotly_white",
        title="Precision vs Recall — All Variants (bubble size = F1)",
        showlegend=False,
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: Impact of Enhancement Strategies
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("3. Impact of Enhancement Strategies")

    st.markdown("""
    We evaluated three progressive enhancement tiers on both detectors:
    - **Baseline** — raw detector at native scale
    - **Multi-Scale** — run inference at scales [0.75×, 1.0×, 1.5×] and merge with Soft-NMS
    - **Tiled + Multi-Scale** — divide large images into overlapping 640×640 tiles, then multi-scale
    """)

    # Build enhancement impact table
    enhancement_data = []
    for variants_group, det_label in [(rf_variants, "RetinaFace"), (mt_variants, "MTCNN")]:
        base_row = variants_group[~variants_group["variant"].str.contains("Multi|Tiled", case=False)]
        ms_row = variants_group[variants_group["variant"].str.contains("MultiScale", case=False) &
                                ~variants_group["variant"].str.contains("Tiled", case=False)]
        tms_row = variants_group[variants_group["variant"].str.contains("Tiled", case=False)]

        for label, row_df in [("Baseline", base_row), ("+ MultiScale", ms_row), ("+ Tiled + MultiScale", tms_row)]:
            if not row_df.empty:
                r = row_df.iloc[0]
                base_f1 = float(base_row.iloc[0]["f1"]) if not base_row.empty else 0
                delta = float(r["f1"]) - base_f1
                enhancement_data.append({
                    "Detector": det_label,
                    "Enhancement": label,
                    "Precision": f"{float(r['precision']):.3f}",
                    "Recall": f"{float(r['recall']):.3f}",
                    "F1": f"{float(r['f1']):.3f}",
                    "F1 Delta": f"+{delta*100:.1f}pp" if delta > 0 else "—",
                    "TP": f"{int(r['tp']):,}",
                    "FP": f"{int(r['fp']):,}",
                })

    enh_df = pd.DataFrame(enhancement_data)
    st.dataframe(enh_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Key observations:**
    - **Tiled inference is the biggest single improvement** — it unlocks detection of small faces
      in high-resolution crowd images by zooming into 640×640 windows.
    - **Multi-Scale alone offers modest gains** — the 1.5× upscale helps medium-small faces but
      the 0.75× downscale adds little value.
    - **RetinaFace benefits more from tiling than MTCNN** — RetinaFace's precision stays above 92%
      even with aggressive tiling, while MTCNN's drops to ~55% due to excessive false positives.
    - **The precision–recall tradeoff is real** — every enhancement that boosts recall also
      introduces more false positives. RetinaFace manages this tradeoff far better.
    """)

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: Root Cause of Detection Failures
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("4. Root Cause Analysis — Why Faces Are Missed")

    # Load quality data for best variant
    best_v = best["variant"]
    q_df = variants[best_v]["quality"]

    if not q_df.empty:
        missed = q_df[q_df["category"] == "undetected"]
        detected = q_df[q_df["category"] == "detected"]
        n_missed = len(missed)
        n_detected = len(detected)

        if n_missed > 0:
            tiny_pct = (missed["face_area"] < 256).sum() / n_missed * 100
            small_pct = (missed["face_area"] < 1024).sum() / n_missed * 100
            med_pct = (missed["face_area"] < 4096).sum() / n_missed * 100
            dark_pct = (missed["face_brightness"] < 80).sum() / n_missed * 100

            st.markdown(f"**Analyzing the {n_missed:,} faces still missed by the best variant ({best_name}):**")

            # Failure breakdown as styled metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""
            <div style="background:#F44336; border-radius:12px; padding:16px; color:white; text-align:center;">
                <h3 style="margin:0; font-size:14px; opacity:0.85;">Smaller than 32×32 px</h3>
                <h1 style="margin:5px 0 0 0; font-size:36px;">{small_pct:.0f}%</h1>
                <p style="margin:0; font-size:12px; opacity:0.7;">{(missed['face_area'] < 1024).sum():,} faces</p>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div style="background:#FF9800; border-radius:12px; padding:16px; color:white; text-align:center;">
                <h3 style="margin:0; font-size:14px; opacity:0.85;">Smaller than 16×16 px</h3>
                <h1 style="margin:5px 0 0 0; font-size:36px;">{tiny_pct:.0f}%</h1>
                <p style="margin:0; font-size:12px; opacity:0.7;">{(missed['face_area'] < 256).sum():,} faces</p>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""
            <div style="background:#9C27B0; border-radius:12px; padding:16px; color:white; text-align:center;">
                <h3 style="margin:0; font-size:14px; opacity:0.85;">Dark Faces (V < 80)</h3>
                <h1 style="margin:5px 0 0 0; font-size:36px;">{dark_pct:.0f}%</h1>
                <p style="margin:0; font-size:12px; opacity:0.7;">{(missed['face_brightness'] < 80).sum():,} faces</p>
            </div>""", unsafe_allow_html=True)
            c4.markdown(f"""
            <div style="background:#607D8B; border-radius:12px; padding:16px; color:white; text-align:center;">
                <h3 style="margin:0; font-size:14px; opacity:0.85;">Mean Missed Area</h3>
                <h1 style="margin:5px 0 0 0; font-size:36px;">{missed['face_area'].mean():.0f}</h1>
                <p style="margin:0; font-size:12px; opacity:0.7;">px² vs {detected['face_area'].mean():,.0f} px² detected</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Failure cause ranking
            st.markdown("#### Failure Cause Hierarchy")

            causes = [
                ("Face too small (< 32×32 px)", small_pct, "#F44336",
                 "The detector's effective receptive field cannot resolve faces below ~30 pixels. "
                 "This is a fundamental resolution limit of the model architecture."),
                (f"Smaller than 64×64 px", med_pct, "#FF5722",
                 "Nearly all missed faces fall below 64×64 pixels, confirming face size as the dominant factor."),
                (f"Dark illumination (V < 80)", dark_pct, "#9C27B0",
                 "About a quarter of missed faces are in poorly-lit regions. CLAHE preprocessing provides partial mitigation."),
            ]

            for cause, pct, color, explanation in causes:
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin:8px 0; padding:10px;
                            border-left:5px solid {color}; background:#fafafa; border-radius:4px;">
                    <div style="min-width:80px; font-size:24px; font-weight:bold; color:{color};">{pct:.0f}%</div>
                    <div>
                        <b>{cause}</b><br>
                        <span style="font-size:13px; color:#666;">{explanation}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: Attribute-Based Findings
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("5. Attribute-Based Findings")

    attr_df = variants[best_v]["attribute"]
    if not attr_df.empty:
        st.markdown(f"**Using the best variant: {best_name}**")

        # Build a consolidated attribute impact table
        attr_summary = []
        for attr in attr_df["attribute"].unique():
            subset = attr_df[attr_df["attribute"] == attr]
            best_a = subset.loc[subset["recall"].idxmax()]
            worst_a = subset.loc[subset["recall"].idxmin()]
            gap = float(best_a["recall"]) - float(worst_a["recall"])
            attr_summary.append({
                "Attribute": attr.capitalize(),
                "Easiest Level": f"{best_a['label']} ({float(best_a['recall']):.3f})",
                "Hardest Level": f"{worst_a['label']} ({float(worst_a['recall']):.3f})",
                "Recall Gap": f"{gap:.3f}",
                "Impact": "Critical" if gap > 0.3 else ("Significant" if gap > 0.15 else "Moderate"),
            })

        attr_impact_df = pd.DataFrame(attr_summary)
        st.dataframe(attr_impact_df, use_container_width=True, hide_index=True)

        # Radar chart of attribute impacts
        categories_radar = [r["Attribute"] for r in attr_summary]
        gaps = [float(r["Recall Gap"]) for r in attr_summary]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=gaps + [gaps[0]],
            theta=categories_radar + [categories_radar[0]],
            fill="toself",
            fillcolor="rgba(233, 69, 96, 0.2)",
            line=dict(color="#e94560", width=2),
            name="Recall Gap (best − worst level)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(gaps) * 1.2]),
            ),
            height=400, template="plotly_white",
            title="Attribute Difficulty Radar — Recall Gap Between Easiest and Hardest Levels",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - **Heavy blur** causes the biggest recall drop (~36pp) — heavily blurred faces lose the edge structure
          that convolutional detectors rely on.
        - **Heavy occlusion** is the second-hardest condition (~45pp gap) — when most of the face is hidden,
          there simply isn't enough visible information for detection.
        - **Extreme illumination** actually *helps* slightly — bright spotlights in events create high-contrast
          faces that are easier to detect than dull, uniform lighting.
        - **Atypical pose** costs ~5pp recall — profile or tilted faces are harder but not catastrophic for
          modern detectors trained on diverse data.
        - **Exaggerated expressions** are actually *easier* to detect — likely because exaggerated faces tend
          to be close-up and well-lit (e.g., performances, celebrations).
        """)

    # ═══════════════════════════════════════════════════════════
    # SECTION 6: Scene & Crowd Insights
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("6. Scene & Crowd Density Insights")

    ev_df = variants[best_v]["event"]
    gw_best = variants[best_v].get("groupwise", pd.DataFrame())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Hardest Scene Categories")
        if not ev_df.empty:
            hard_5 = ev_df.sort_values("recall").head(5)
            for _, row in hard_5.iterrows():
                color = "#F44336" if row["recall"] < 0.6 else "#FF9800"
                st.markdown(f"- **{row['event']}**: recall = `{row['recall']:.3f}` "
                            f"({int(row['num_images'])} images, {int(row['total_gt']):,} faces)")
            st.caption("Low recall in Matador/Bullfighter and Basketball scenes — "
                       "these involve dense distant crowds where most faces are < 20×20 px.")

    with c2:
        st.markdown("#### Crowd Density Degradation")
        if not gw_best.empty:
            sparse = gw_best[gw_best["bin"] == "0-10"]
            dense = gw_best[gw_best["bin"] == "51+"]
            if not sparse.empty and not dense.empty:
                sparse_r = float(sparse.iloc[0]["recall"])
                dense_r = float(dense.iloc[0]["recall"])
                drop = sparse_r - dense_r
                st.metric("Recall (0–10 faces)", f"{sparse_r:.3f}", help="Sparse images")
                st.metric("Recall (51+ faces)", f"{dense_r:.3f}",
                          delta=f"-{drop*100:.1f}pp", delta_color="inverse", help="Dense crowd images")
                st.caption(f"Recall drops by **{drop*100:.1f} percentage points** as images go from sparse to "
                           f"densely crowded — dense images contain primarily tiny, overlapping faces.")

    # ═══════════════════════════════════════════════════════════
    # SECTION 7: Precision-Recall Tradeoff
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("7. Optimal Operating Point")

    thresh_df = variants[best_v]["threshold"]
    if not thresh_df.empty:
        best_thresh_idx = thresh_df["f1"].idxmax()
        best_thresh = thresh_df.iloc[best_thresh_idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Threshold", f"{float(best_thresh['threshold']):.1f}")
        c2.metric("F1 at Optimal", f"{float(best_thresh['f1']):.3f}")
        c3.metric("Precision at Optimal", f"{float(best_thresh['precision']):.3f}")
        c4.metric("Recall at Optimal", f"{float(best_thresh['recall']):.3f}")

        st.markdown(f"""
        The **F1-optimal confidence threshold is {float(best_thresh['threshold']):.1f}** for {best_name}.
        Lowering to 0.3 doesn't change results (RetinaFace is already highly confident in its detections),
        while raising to 0.6+ causes a sharp recall collapse because the detector becomes overly selective.

        **Practical recommendation:** Use threshold **0.4–0.5** for balanced operation. For privacy-critical
        applications (anonymization), use **0.3** to maximize recall at the cost of a few extra false positives.
        For surveillance with low false-alarm tolerance, use **0.7+**.
        """)

    # ═══════════════════════════════════════════════════════════
    # SECTION 8: Actionable Recommendations
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("8. Actionable Recommendations")

    recs = [
        ("Use RetinaFace + Tiled + Multi-Scale as the production detector",
         "It achieves the best F1 and recall while maintaining >92% precision. "
         "The ONNX Runtime backend enables GPU acceleration without a full deep learning framework.",
         "✅", "#4CAF50"),
        ("Deploy on GPU for real-time performance",
         "RetinaFace via ONNX CUDAExecutionProvider runs on GPU natively. "
         "MTCNN requires CUDA-enabled PyTorch. For inference-only deployments, ONNX is lighter.",
         "⚡", "#2196F3"),
        ("Accept the sub-32px face limitation",
         "94% of missed faces are smaller than 32×32 pixels — this is a fundamental resolution limit. "
         "Super-resolution preprocessing or purpose-built tiny-face detectors (S3FD, DSFD) would be needed.",
         "⚠️", "#FF9800"),
        ("Use CLAHE preprocessing for dark environments",
         "26% of missed faces are in dark regions. Adaptive CLAHE on the L channel improves contrast "
         "without affecting already well-lit faces (adaptive mode).",
         "🔧", "#9C27B0"),
        ("Lower confidence to 0.3 for anonymization pipelines",
         "For face anonymization (where missing a face is worse than a false blur), "
         "use threshold 0.3 — precision stays above 92% while recall improves.",
         "🎯", "#e94560"),
        ("Consider ensemble for critical applications",
         "RetinaFace and MTCNN have complementary strengths — MTCNN occasionally catches faces RetinaFace misses. "
         "An ensemble with Soft-NMS merging can recover 1–3% additional recall.",
         "🔀", "#607D8B"),
    ]

    for title, body, icon, color in recs:
        st.markdown(f"""
        <div style="border-left:5px solid {color}; padding:12px 16px; margin:10px 0;
                    background:#fafafa; border-radius:0 8px 8px 0;">
            <span style="font-size:20px;">{icon}</span>
            <b style="font-size:15px; margin-left:8px;">{title}</b><br>
            <span style="font-size:13px; color:#555; line-height:1.6;">{body}</span>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SECTION 9: AdaSR-Face — Super-Resolution for Missed Faces
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("9. AdaSR-Face — Adaptive Super-Resolution for Missed Faces")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #2d1b4e 50%, #4a1942 100%);
                border-radius: 16px; padding: 24px; color: white; margin: 15px 0;">
        <h3 style="margin-top:0; color: #bb86fc;">Novel Contribution: AdaSR-Face Pipeline</h3>
        <p style="font-size:15px; line-height:1.7;">
            Since <b>94% of missed faces are sub-32px</b>, we propose <b>AdaSR-Face</b>
            (Adaptive Super-Resolution guided by Detection Confidence) — a <b>two-stage cascade</b> that
            selectively applies Real-ESRGAN super-resolution only to image regions where the detector
            is least confident, then re-detects and fuses results via Soft-NMS.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### How AdaSR-Face Works")
    st.markdown("""
    1. **Stage 1 — Initial Detection:** Run SCRFD at standard resolution to get detections + confidence scores
    2. **Identify Weak Regions:** Find image patches where confidence is low or where small faces cluster
    3. **Selective SR:** Apply Real-ESRGAN 2× upscaling **only** to those weak patches (not the whole image)
    4. **Stage 2 — Re-detection:** Run the detector again on the super-resolved patches
    5. **Soft-NMS Fusion:** Merge Stage 1 and Stage 2 detections, suppressing duplicates while keeping new finds
    """)

    # Load ablation data if available
    _adasr_summary_path = EXP_CSV_DIR / "ablation_summary.csv"
    _adasr_cascade_path = EXP_CSV_DIR / "ablation_cascade_stats_E5_scrfd_adaptive_sr.csv"
    _adasr_avail = _adasr_summary_path.exists()

    if _adasr_avail:
        _abl_df = pd.read_csv(_adasr_summary_path)

        # Get key experiments
        _e1 = _abl_df[_abl_df["experiment"].str.contains("E1", na=False)]
        _e2 = _abl_df[_abl_df["experiment"].str.contains("E2", na=False)]
        _e3 = _abl_df[_abl_df["experiment"].str.contains("E3", na=False)]
        _e4 = _abl_df[_abl_df["experiment"].str.contains("E4", na=False)]
        _e5 = _abl_df[_abl_df["experiment"].str.contains("E5", na=False)]

        st.markdown("#### Ablation Study Results")

        if not _e1.empty and not _e5.empty:
            _e1_f1 = float(_e1.iloc[0]["f1"])
            _e5_f1 = float(_e5.iloc[0]["f1"])
            _e5_r = float(_e5.iloc[0]["recall"])
            _e5_p = float(_e5.iloc[0]["precision"])
            _e1_r = float(_e1.iloc[0]["recall"])
            _delta_r = (_e5_r - _e1_r) * 100

            # Key metrics callout
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""
            <div style="background:#bb86fc; border-radius:12px; padding:14px; color:white; text-align:center;">
                <h3 style="margin:0; font-size:13px; opacity:0.85;">AdaSR F1</h3>
                <h1 style="margin:4px 0 0 0; font-size:32px;">{_e5_f1:.3f}</h1>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div style="background:#03dac6; border-radius:12px; padding:14px; color:#1a1a2e; text-align:center;">
                <h3 style="margin:0; font-size:13px; opacity:0.85;">Recall Gain vs Baseline</h3>
                <h1 style="margin:4px 0 0 0; font-size:32px;">+{_delta_r:.1f}pp</h1>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""
            <div style="background:#cf6679; border-radius:12px; padding:14px; color:white; text-align:center;">
                <h3 style="margin:0; font-size:13px; opacity:0.85;">New Faces Found</h3>
                <h1 style="margin:4px 0 0 0; font-size:32px;">{int(_e5.iloc[0]['tp']) - int(_e1.iloc[0]['tp']) if int(_e5.iloc[0]['total_gt']) >= int(_e1.iloc[0]['total_gt']) else '—'}</h1>
            </div>""", unsafe_allow_html=True)
            c4.markdown(f"""
            <div style="background:#3700b3; border-radius:12px; padding:14px; color:white; text-align:center;">
                <h3 style="margin:0; font-size:13px; opacity:0.85;">Precision</h3>
                <h1 style="margin:4px 0 0 0; font-size:32px;">{_e5_p:.3f}</h1>
            </div>""", unsafe_allow_html=True)

        # Full ablation comparison table
        _abl_display = []
        for _, _r in _abl_df.iterrows():
            _abl_display.append({
                "Experiment": _r["experiment"],
                "Method": _r["description"][:60] + "…" if len(str(_r["description"])) > 60 else _r["description"],
                "Precision": f"{float(_r['precision']):.3f}",
                "Recall": f"{float(_r['recall']):.3f}",
                "F1": f"{float(_r['f1']):.3f}",
                "TP": int(_r["tp"]),
            })
        st.dataframe(pd.DataFrame(_abl_display), use_container_width=True, hide_index=True)

        # Cascade stats if available
        if _adasr_cascade_path.exists():
            _cas_df = pd.read_csv(_adasr_cascade_path)
            total_new = int(_cas_df["stage2_new_faces"].sum())
            total_sr_regions = int(_cas_df["sr_regions"].sum())
            st.markdown(f"""
            **Cascade efficiency:** Across the test set, AdaSR-Face processed **{total_sr_regions} SR regions**
            and recovered **{total_new} previously-undetected faces** in Stage 2 re-detection.
            The selective approach avoids wasting compute on already-confident regions.
            """)

        st.markdown("#### Key Findings from the Ablation")
        st.markdown("""
        - **Tiled + Multi-Scale (E2)** remains the single biggest accuracy boost — it improves F1 by
          ~+16pp over the SCRFD baseline through spatial windowing
        - **Blind full-image SR (E4)** actually *hurts* performance — upscaling the entire image
          introduces artifacts and wastes compute on already-detectable faces
        - **AdaSR-Face (E5)** recovers faces that tiling alone misses by targeting only the sub-32px
          regions where the detector struggles, yielding a **+11pp recall lift over baseline**
        - **Selective SR is the key insight** — applying SR everywhere is counterproductive, but
          applying it *only where detection confidence is low* provides a clean recall boost
        """)

    else:
        st.info("Run the ablation experiments (`python experiments/run_ablation.py`) to see AdaSR-Face results here.")

    st.markdown("""
    <div style="border-left:5px solid #bb86fc; padding:12px 16px; margin:15px 0;
                background:#1a1a2e; border-radius:0 8px 8px 0; color:white;">
        <b style="font-size:15px;">Research Significance</b><br>
        <span style="font-size:13px; line-height:1.6;">
        AdaSR-Face demonstrates that <b>targeted super-resolution guided by detection confidence</b>
        is more effective than blind upscaling. By concentrating compute on the hardest patches,
        the pipeline achieves meaningful recall gains on sub-32px faces — the dominant failure mode
        identified in our analysis — while maintaining practical inference speeds.
        This confidence-guided selective processing paradigm generalizes beyond face detection
        to any object detection task where small targets dominate the failure distribution.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SECTION 10: Limitations & Future Work
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("10. Limitations & Future Work")

    st.markdown("""
    | Limitation | Impact | Potential Mitigation |
    |:-----------|:-------|:--------------------|
    | **No custom training** — both models use pre-trained weights | Models are not fine-tuned to the target domain | Fine-tune on domain-specific data with transfer learning |
    | **WIDER FACE only** — all evaluation on one dataset | May not generalize to surveillance, selfies, medical | Evaluate on FDDB, AFW, MAFA (masked faces), and real-world data |
    | **CPU-only MTCNN** — PyTorch CUDA was not installed | MTCNN runs ~5× slower than possible | Install `torch` with CUDA support for fair speed comparison |
    | **No speed benchmarking** — only accuracy is measured | Cannot recommend for real-time vs. batch use cases | Add FPS measurement per variant, compare GPU vs CPU throughput |
    | **Fixed IoU threshold (0.5)** — standard but arbitrary | Loose boxes that overlap >40% are counted as misses | Evaluate at IoU 0.3, 0.5, 0.75 (COCO-style mAP) |
    | **No temporal / video evaluation** — image-only | Face tracking and temporal consistency not measured | Extend to video datasets with tracking metrics (MOTA, IDF1) |
    """)

    # ═══════════════════════════════════════════════════════════
    # SECTION 11: Final Verdict
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
                border-radius:16px; padding:30px; color:white; text-align:center; margin:20px 0;">
        <h2 style="margin:0; color:#38ef7d;">Final Verdict</h2>
        <p style="font-size:18px; margin:15px 0;">
            <b>RetinaFace + Tiled + Multi-Scale</b> is the recommended production configuration,
            achieving <b style="color:#38ef7d;">{float(rf_best['f1']):.1%} F1</b> with
            <b style="color:#21CBF3;">{float(rf_best['precision']):.1%} precision</b>,
            finding <b style="color:#f2c94c;">{int(rf_best['tp']):,}</b> of {int(rf_best['total_gt']):,} faces.
        </p>
        <p style="font-size:16px; margin:10px 0;">
            For the hardest sub-32px faces, <b style="color:#bb86fc;">AdaSR-Face</b> provides an additional
            <b style="color:#bb86fc;">+11pp recall boost</b> through confidence-guided selective super-resolution
            — demonstrating that <em>where</em> you apply SR matters more than <em>whether</em> you apply it.
        </p>
        <p style="font-size:14px; opacity:0.7; margin:10px 0 0 0;">
            Evaluated on WIDER FACE validation set · 3,226 images · 39,123 valid GT faces · 6 detector variants + 5 ablation experiments
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Appendix (Plot Gallery + Full Report)
# ══════════════════════════════════════════════════════════════
elif page == "📎 Appendix":
    st.title("📎 Appendix — Plots & Full Report")
    st.markdown("Pre-rendered matplotlib figures for papers/presentations, plus the full text analysis report.")

    _app_tab = st.radio("Section", ["🖼️ Plot Gallery", "📄 Full Text Report"], horizontal=True, key="appendix_tab")

    if _app_tab == "🖼️ Plot Gallery":
        st.subheader("🖼️ Plot Gallery")
        plot_files = sorted(PLOTS_DIR.glob("*.png"))
        if not plot_files:
            st.warning("No plots found in outputs/plots/. Run: `python scripts/generate_report.py`")
        else:
            categories = {
                "Metrics Comparison": [], "Group-wise Analysis": [],
                "Quality Distributions (Box Plots)": [], "Face Size Histograms": [],
                "Brightness vs. Size (Scatter)": [], "Confidence Distribution": [],
                "Distance Histograms": [], "Attribute-Based Recall": [],
                "Event/Scene Category": [], "Precision-Recall Curve": [],
                "Threshold Sensitivity": [],
            }
            for pf in plot_files:
                name = pf.stem.lower()
                if "metrics_comparison" in name: categories["Metrics Comparison"].append(pf)
                elif "groupwise" in name: categories["Group-wise Analysis"].append(pf)
                elif "quality_distributions" in name: categories["Quality Distributions (Box Plots)"].append(pf)
                elif "face_size" in name: categories["Face Size Histograms"].append(pf)
                elif "brightness_vs_size" in name: categories["Brightness vs. Size (Scatter)"].append(pf)
                elif "confidence" in name: categories["Confidence Distribution"].append(pf)
                elif "distance" in name: categories["Distance Histograms"].append(pf)
                elif "attribute" in name: categories["Attribute-Based Recall"].append(pf)
                elif "event" in name: categories["Event/Scene Category"].append(pf)
                elif "pr_curve" in name: categories["Precision-Recall Curve"].append(pf)
                elif "threshold" in name: categories["Threshold Sensitivity"].append(pf)

            for cat_name, files in categories.items():
                if not files:
                    continue
                st.markdown(f"**{cat_name}**")
                cols = st.columns(min(len(files), 2))
                for i, pf in enumerate(files):
                    with cols[i % len(cols)]:
                        st.image(str(pf), caption=pf.stem, use_container_width=True)

    elif _app_tab == "📄 Full Text Report":
        st.subheader("📄 Full Analysis Report")
        report_text, report_name = load_report_text()
        if not report_text:
            st.warning("No report found. Run: `python scripts/generate_report.py`")
        else:
            st.caption(f"Source: `{report_name}`")
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report_text,
                file_name=report_name,
                mime="text/plain",
            )
            st.code(report_text, language=None)
