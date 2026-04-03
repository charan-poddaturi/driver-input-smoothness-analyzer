"""
DISA — Streamlit Dashboard
============================
Five-tab research dashboard for the Driver Input Smoothness Analyzer.

Tabs:
  1. Upload & Analyze    — Upload CSV → full score breakdown
  2. Simulator           — Generate and download synthetic datasets
  3. Weight Optimizer    — Tune weights from labeled data
  4. Validation          — Run research validation suite
  5. Driver Comparison   — Rank multiple sessions

Run with: streamlit run frontend/app.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Internal imports
from backend.preprocessing import preprocess, preprocess_for_features, SIGNAL_COLUMNS
from backend.features import extract_features, get_event_highlight_mask
from backend.scoring import compute_smoothness_score, compute_timeseries_score, rank_sessions
from backend.anomaly import detect_all_anomalies
from backend.optimizer import prepare_training_data, optimize_weights
from validation.validate import run_validation, hierarchy_test

# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DISA — Driver Input Smoothness Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0f1e 0%, #12122a 50%, #0a0a1a 100%);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #a0a0c0;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Score badge */
.score-badge-A { color: #34d399; font-weight: 700; font-size: 1.6rem; }
.score-badge-B { color: #6ee7b7; font-weight: 700; font-size: 1.6rem; }
.score-badge-C { color: #fbbf24; font-weight: 700; font-size: 1.6rem; }
.score-badge-D { color: #f97316; font-weight: 700; font-size: 1.6rem; }
.score-badge-F { color: #ef4444; font-weight: 700; font-size: 1.6rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c7c7e0;
    border-left: 3px solid #6366f1;
    padding-left: 0.7rem;
    margin: 1.5rem 0 0.8rem 0;
}

/* Info box */
.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #b0b0d0;
    margin: 0.5rem 0;
}

/* Anomaly tag */
.event-tag {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #fca5a5;
    margin: 2px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,15,30,0.95);
    border-right: 1px solid rgba(99,102,241,0.15);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,12,28,0.8)",
    font=dict(color="#c0c0d8", family="Inter"),
    xaxis=dict(gridcolor="rgba(99,102,241,0.12)", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(99,102,241,0.12)", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
)

SIGNAL_COLORS = {
    "steering_angle": "#6366f1",
    "throttle_position": "#34d399",
    "brake_pressure": "#ef4444",
    "acceleration_x": "#fbbf24",
    "acceleration_y": "#f97316",
    "acceleration_z": "#a78bfa",
    "gyroscope_yaw_rate": "#38bdf8",
}


def _score_color(score: float) -> str:
    if score >= 85:
        return "#34d399"
    if score >= 70:
        return "#6ee7b7"
    if score >= 55:
        return "#fbbf24"
    if score >= 40:
        return "#f97316"
    return "#ef4444"


def _grade_badge(grade: str) -> str:
    letter = grade[0] if grade else "?"
    css_class = f"score-badge-{letter}"
    return f'<span class="{css_class}">{grade}</span>'


def _load_and_validate_csv(uploaded) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: `{missing}`")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Failed to parse CSV: {e}")
        return None


def _radar_chart(sub_scores: dict) -> go.Figure:
    labels = [k.replace("_", " ").title() for k in sub_scores]
    values = list(sub_scores.values())
    labels += [labels[0]]
    values += [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        fillcolor="rgba(99,102,241,0.25)",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(color="#a78bfa", size=7),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(12,12,28,0.8)",
            radialaxis=dict(range=[0, 100], gridcolor="rgba(99,102,241,0.2)",
                            tickfont=dict(color="#888", size=9)),
            angularaxis=dict(gridcolor="rgba(99,102,241,0.2)",
                             tickfont=dict(color="#c0c0d8", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
    )
    return fig


def _timeseries_plot(df: pd.DataFrame, channels: list, highlight_mask=None, anomalies=None) -> go.Figure:
    fig = go.Figure()
    ts = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df)) / 25.0

    for ch in channels:
        if ch not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=ts, y=df[ch].values,
            name=ch.replace("_", " ").title(),
            line=dict(color=SIGNAL_COLORS.get(ch, "#888"), width=1.5),
            opacity=0.9,
        ))

    # Highlight anomaly regions
    if anomalies:
        for ev in anomalies:
            fig.add_vrect(
                x0=ev.start_time, x1=ev.end_time,
                fillcolor="rgba(239,68,68,0.12)",
                line=dict(color="rgba(239,68,68,0.5)", width=1, dash="dot"),
                annotation_text=ev.event_type.replace("_", " "),
                annotation_font=dict(size=8, color="#fca5a5"),
                annotation_position="top left",
            )

    fig.update_layout(**PLOTLY_LAYOUT, height=320)
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Signal Value")
    return fig


def _contribution_bar(contributions: dict) -> go.Figure:
    labels = list(contributions.keys())
    values = list(contributions.values())
    colors = ["#6366f1", "#34d399", "#fbbf24", "#ef4444"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors[:len(labels)],
        marker_line_width=0,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
    fig.update_yaxes(title="Penalty Points")
    fig.update_xaxes(title="Feature")
    return fig


def _sensitivity_plot(sensitivity: dict, deltas=None) -> go.Figure:
    if deltas is None:
        deltas = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

    fig = go.Figure()
    colors_s = {"w_S": "#6366f1", "w_J": "#34d399", "w_V": "#fbbf24", "w_P": "#ef4444"}

    for name, mse_vals in sensitivity.items():
        fig.add_trace(go.Scatter(
            x=deltas, y=mse_vals,
            name=name,
            mode="lines+markers",
            line=dict(color=colors_s.get(name, "#888"), width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    fig.update_xaxes(title="Weight Perturbation (Δw)")
    fig.update_yaxes(title="MSE")
    return fig


def _robustness_plot(robustness: dict) -> go.Figure:
    fig = go.Figure()
    colors_r = ["#6366f1", "#34d399", "#fbbf24"]

    for i, (profile, data) in enumerate(robustness.items()):
        xs = [float(k) for k in data.keys()]
        ys = [v if v is not None else float("nan") for v in data.values()]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=profile,
            mode="lines+markers",
            line=dict(color=colors_r[i % 3], width=2),
        ))

    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    fig.update_xaxes(title="Noise Level (σ_frac)", type="log")
    fig.update_yaxes(title="Smoothness Score", range=[0, 100])
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size: 2rem;'>🚗</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #a78bfa; margin-top: 0.3rem;'>DISA</div>
        <div style='font-size: 0.72rem; color: #6060a0;'>Driver Input Smoothness Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**⚙️ Global Settings**")
    fs_global = st.number_input("Sample Rate (Hz)", min_value=1.0, max_value=1000.0, value=25.0, step=5.0)

    st.markdown("**🎚️ Scoring Weights**")
    st.caption("Must sum to 1.0")
    w_S = st.slider("w₁ · Rate of Change (S)", 0.0, 1.0, 0.25, 0.01)
    w_J = st.slider("w₂ · Jerk (J)", 0.0, 1.0, 0.30, 0.01)
    w_V = st.slider("w₃ · Variance (V)", 0.0, 1.0, 0.25, 0.01)
    w_P = st.slider("w₄ · Spike Freq (P)", 0.0, 1.0, 0.20, 0.01)

    w_total = w_S + w_J + w_V + w_P
    if abs(w_total - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {w_total:.2f} (will be auto-normalized)")

    custom_weights = np.array([w_S, w_J, w_V, w_P])
    custom_weights = custom_weights / custom_weights.sum()

    st.divider()
    st.caption("**Formula:**")
    st.latex(r"\text{Score} = 100 - \sum_{i} w_i \cdot f_i")
    st.divider()
    st.caption("v1.0 · Research MVP · 2024")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Upload & Analyze",
    "🚗 Simulator",
    "🎯 Weight Optimizer",
    "🧪 Validation",
    "🏆 Driver Comparison",
])


# ===========================================================================
# Tab 1: Upload & Analyze
# ===========================================================================

with tab1:
    st.markdown("## 📊 Upload & Analyze")
    st.markdown('<div class="info-box">Upload a CSV file containing time-series driving data. '
                'Required columns: <code>timestamp, steering_angle, throttle_position, brake_pressure, '
                'acceleration_x, acceleration_y, acceleration_z, gyroscope_yaw_rate</code></div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="analyze_upload")

    col_hint, col_gen = st.columns([3, 1])
    with col_gen:
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            from simulator.simulator import generate_smooth_driver
            df_sample, _ = generate_smooth_driver(duration_s=60, fs=fs_global, seed=42)
            csv_bytes = df_sample.to_csv(index=False).encode()
            st.download_button("⬇️ Download Sample CSV", data=csv_bytes,
                               file_name="sample_smooth.csv", mime="text/csv",
                               use_container_width=True)

    if uploaded:
        df_raw = _load_and_validate_csv(uploaded)
        if df_raw is not None:
            with st.spinner("⚙️ Running analysis pipeline..."):
                # Feature extraction on raw (filtered, un-normalized) signals
                feat_df = preprocess_for_features(df_raw, fs=fs_global)
                fm = extract_features(feat_df, fs=fs_global)
                result = compute_smoothness_score(fm, weights=custom_weights)

                # Normalized data for display
                processed_df, norm_stats = preprocess(df_raw, fs=fs_global)

                # Time-series score
                ts_scores = compute_timeseries_score(feat_df, fs=fs_global, weights=result.weights)

                # Anomalies (on raw filtered data)
                anomalies = detect_all_anomalies(feat_df, fs=fs_global)
                raw_df_notnorm = feat_df   # reuse

            # --- Score banner ---
            st.markdown("---")
            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                color = _score_color(result.overall_score)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="background:linear-gradient(135deg,{color},{color}aa);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                        {result.overall_score:.1f}
                    </div>
                    <div class="metric-label">Overall Score</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.6rem;font-weight:700;color:{color};">{result.grade}</div>
                    <div class="metric-label">Grade</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                n_samples = len(df_raw)
                dur = n_samples / fs_global
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dur:.0f}s</div>
                    <div class="metric-label">Session Duration</div>
                </div>""", unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(anomalies)}</div>
                    <div class="metric-label">Rough Events</div>
                </div>""", unsafe_allow_html=True)

            with c5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{n_samples:,}</div>
                    <div class="metric-label">Data Points</div>
                </div>""", unsafe_allow_html=True)

            # --- Sub-scores & Radar ---
            st.markdown('<div class="section-header">Sub-Score Breakdown</div>', unsafe_allow_html=True)
            col_radar, col_sub = st.columns([1, 1])

            with col_radar:
                st.plotly_chart(_radar_chart(result.sub_scores), use_container_width=True)

            with col_sub:
                for group, score in result.sub_scores.items():
                    c_score = _score_color(score)
                    st.markdown(
                        f"**{group.replace('_', ' ').title()}**",
                    )
                    st.progress(int(score), text=f"{score:.1f}/100")

            # --- Feature contributions ---
            st.markdown('<div class="section-header">Feature Penalty Contributions</div>', unsafe_allow_html=True)
            st.plotly_chart(_contribution_bar(result.feature_contributions), use_container_width=True)

            # --- Rolling score timeline ---
            st.markdown('<div class="section-header">Rolling Smoothness Score Over Time</div>', unsafe_allow_html=True)
            ts_arr = df_raw["timestamp"].values if "timestamp" in df_raw.columns else np.arange(len(df_raw)) / fs_global

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts_arr, y=ts_scores,
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="#6366f1", width=2),
                name="Smoothness Score",
            ))
            fig_ts.add_hline(y=70, line_dash="dash", line_color="rgba(251,191,36,0.5)",
                              annotation_text="Good threshold", annotation_font_color="#fbbf24")
            for ev in anomalies:
                fig_ts.add_vrect(
                    x0=ev.start_time, x1=ev.end_time,
                    fillcolor="rgba(239,68,68,0.1)",
                    line=dict(color="rgba(239,68,68,0.4)", width=1),
                )
            fig_ts.update_layout(**PLOTLY_LAYOUT, height=220)
            fig_ts.update_yaxes(range=[0, 100], title="Score")
            fig_ts.update_xaxes(title="Time (s)")
            st.plotly_chart(fig_ts, use_container_width=True)

            # --- Signal plots ---
            st.markdown('<div class="section-header">Signal Time-Series</div>', unsafe_allow_html=True)
            channel_groups = {
                "Steering & Gyro": ["steering_angle", "gyroscope_yaw_rate"],
                "Throttle & Brake": ["throttle_position", "brake_pressure"],
                "Acceleration": ["acceleration_x", "acceleration_y", "acceleration_z"],
            }

            for group_name, channels in channel_groups.items():
                with st.expander(f"📈 {group_name}", expanded=(group_name == "Steering & Gyro")):
                    fig = _timeseries_plot(raw_df_notnorm, channels, anomalies=anomalies)
                    st.plotly_chart(fig, use_container_width=True)

            # --- Anomaly events table ---
            st.markdown('<div class="section-header">Detected Rough Events</div>', unsafe_allow_html=True)
            if anomalies:
                ev_df = pd.DataFrame([ev.to_dict() for ev in anomalies])
                st.dataframe(
                    ev_df.style.background_gradient(subset=["severity"], cmap="RdYlGn_r"),
                    use_container_width=True, height=250
                )
            else:
                st.success("✅ No significant rough events detected — very smooth drive!")

            # --- PDF Download ---
            st.markdown("---")
            if st.button("📄 Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF..."):
                    try:
                        from backend.reports import generate_report
                        pdf_bytes = generate_report(result, anomalies, session_id=uploaded.name)
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"disa_report_{uploaded.name}.pdf",
                            mime="application/pdf",
                        )
                    except ImportError:
                        st.warning("Install fpdf2 for PDF export: `pip install fpdf2`")


# ===========================================================================
# Tab 2: Simulator
# ===========================================================================

with tab2:
    st.markdown("## 🚗 Synthetic Driving Simulator")
    st.markdown('<div class="info-box">Generates physically plausible driving data using Ornstein–Uhlenbeck '
                'processes, sinusoidal waveforms, and event injection (braking, lane changes, stop-and-go). '
                'No random noise — temporally continuous and reproducible.</div>',
                unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        profile = st.selectbox("Driver Profile", ["smooth", "aggressive", "mixed"], index=0)
    with col_s2:
        duration = st.slider("Duration (seconds)", 10, 300, 120, 10)
    with col_s3:
        seed = st.number_input("Random Seed", value=42, min_value=0)

    if st.button("▶️ Generate Dataset", type="primary", use_container_width=True):
        with st.spinner(f"Simulating {profile} driver for {duration}s..."):
            from simulator.simulator import generate_smooth_driver, generate_aggressive_driver, generate_mixed_driver

            generators = {
                "smooth": lambda: generate_smooth_driver(duration_s=duration, fs=fs_global, seed=seed),
                "aggressive": lambda: generate_aggressive_driver(duration_s=duration, fs=fs_global, seed=seed),
                "mixed": lambda: generate_mixed_driver(duration_s=duration, fs=fs_global, seed=seed),
            }
            df_sim, events_sim = generators[profile]()

            st.session_state["sim_df"] = df_sim
            st.session_state["sim_events"] = events_sim
            st.session_state["sim_profile"] = profile

    if "sim_df" in st.session_state:
        df_sim = st.session_state["sim_df"]
        events_sim = st.session_state["sim_events"]
        profile_name = st.session_state["sim_profile"]

        # Quick analysis
        feat_s = preprocess_for_features(df_sim, fs=fs_global)
        fm_s = extract_features(feat_s, fs=fs_global)
        result_s = compute_smoothness_score(fm_s, weights=custom_weights)

        # Stats banner
        c1, c2, c3, c4 = st.columns(4)
        color_s = _score_color(result_s.overall_score)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color_s};">{result_s.overall_score:.1f}</div>
                <div class="metric-label">Predicted Score</div></div>""", unsafe_allow_html=True)
        with c2:
            exp_score = df_sim["expected_score"].mean()
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{exp_score:.1f}</div>
                <div class="metric-label">Expected Score</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(df_sim):,}</div>
                <div class="metric-label">Samples</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(events_sim)}</div>
                <div class="metric-label">Events Injected</div></div>""", unsafe_allow_html=True)

        # Timeseries
        st.markdown('<div class="section-header">Simulated Signals</div>', unsafe_allow_html=True)
        channel_view = st.multiselect(
            "Select channels to display",
            options=SIGNAL_COLUMNS,
            default=["steering_angle", "throttle_position", "brake_pressure"],
            key="sim_channels",
        )
        if channel_view:
            event_objs = [type("E", (), {"start_time": e.start_idx / fs_global,
                                          "end_time": e.end_idx / fs_global,
                                          "event_type": e.name})()
                          for e in events_sim]
            fig_sim = _timeseries_plot(df_sim, channel_view, anomalies=event_objs)
            st.plotly_chart(fig_sim, use_container_width=True)

        # Events table
        st.markdown('<div class="section-header">Injected Events</div>', unsafe_allow_html=True)
        if events_sim:
            ev_data = [{
                "Event": e.name,
                "Start (s)": round(e.start_idx / fs_global, 2),
                "End (s)": round(e.end_idx / fs_global, 2),
                "Severity": round(e.severity, 2),
            } for e in events_sim]
            st.dataframe(pd.DataFrame(ev_data), use_container_width=True, height=200)

        # Download
        csv_bytes = df_sim.to_csv(index=False).encode()
        st.download_button(
            label=f"⬇️ Download {profile_name}.csv",
            data=csv_bytes,
            file_name=f"{profile_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Generate all three
        st.markdown("---")
        if st.button("📦 Generate All Three Profiles (smooth + aggressive + mixed)"):
            from simulator.simulator import generate_all_datasets
            with st.spinner("Generating all datasets..."):
                generate_all_datasets(output_dir=str(ROOT / "data"), fs=fs_global)
            st.success("✅ Datasets saved to `data/` directory!")


# ===========================================================================
# Tab 3: Weight Optimizer
# ===========================================================================

with tab3:
    st.markdown("## 🎯 Data-Driven Weight Optimizer")
    st.markdown('<div class="info-box">Upload labeled CSV files (with <code>expected_score</code> column) '
                'to optimize the scoring weights using SLSQP constrained optimization and/or grid search. '
                'Constraint: w₁ + w₂ + w₃ + w₄ = 1.</div>', unsafe_allow_html=True)

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        opt_method = st.selectbox("Optimization Method", ["slsqp", "grid", "both"], index=0)
    with col_opt2:
        opt_seg = st.slider("Segment Length (s)", 5, 30, 10)
    with col_opt3:
        opt_restarts = st.number_input("SLSQP Restarts", min_value=1, max_value=20, value=5)

    opt_files = st.file_uploader("Upload labeled CSV files",
                                 type=["csv"], accept_multiple_files=True, key="opt_upload")

    if opt_files and st.button("▶️ Run Optimization", type="primary"):
        import tempfile, os
        tmp_paths = []
        for f in opt_files:
            contents = f.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(contents)
                tmp_paths.append(tmp.name)

        with st.spinner("Extracting features and optimizing weights..."):
            try:
                feat_list, exp_scores = prepare_training_data(tmp_paths, fs=fs_global,
                                                               segment_length_s=opt_seg)
                if len(feat_list) < 3:
                    st.error(f"Not enough segments ({len(feat_list)}). Upload more/longer files.")
                else:
                    opt_result = optimize_weights(feat_list, exp_scores, method=opt_method,
                                                  n_restarts=int(opt_restarts))

                    st.session_state["opt_result"] = opt_result
            except Exception as e:
                st.error(f"Optimization failed: {e}")
            finally:
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    if "opt_result" in st.session_state:
        opt_r = st.session_state["opt_result"]

        st.markdown('<div class="section-header">Optimization Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        w_opt = opt_r.optimized_weights

        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{opt_r.mse:.3f}</div>
                <div class="metric-label">Optimized MSE</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{opt_r.mae:.3f}</div>
                <div class="metric-label">Optimized MAE</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{opt_r.mse_improvement_pct:.1f}%</div>
                <div class="metric-label">MSE Improvement</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{opt_r.method.split()[0]}</div>
                <div class="metric-label">Method</div></div>""", unsafe_allow_html=True)

        # Weight comparison
        st.markdown('<div class="section-header">Weight Comparison</div>', unsafe_allow_html=True)
        weight_labels = ["w₁ (S)", "w₂ (J)", "w₃ (V)", "w₄ (P)"]
        default_w = [0.25, 0.30, 0.25, 0.20]

        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(name="Default", x=weight_labels, y=default_w,
                               marker_color="rgba(99,102,241,0.5)"))
        fig_w.add_trace(go.Bar(name="Optimized", x=weight_labels, y=w_opt.tolist(),
                               marker_color="#6366f1"))
        fig_w.update_layout(**PLOTLY_LAYOUT, height=280, barmode="group",
                             title="Default vs Optimized Weights")
        st.plotly_chart(fig_w, use_container_width=True)

        # Sensitivity analysis
        st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
        st.caption("Shows how MSE changes when each weight is perturbed by ±Δw")
        fig_sens = _sensitivity_plot(opt_r.sensitivity)
        st.plotly_chart(fig_sens, use_container_width=True)

        # Copy weights button
        weight_json = json.dumps({k: round(float(v), 4)
                                  for k, v in zip(["w_S", "w_J", "w_V", "w_P"], w_opt.tolist())},
                                 indent=2)
        st.code(weight_json, language="json")
        st.download_button("⬇️ Download Weights JSON", data=weight_json,
                           file_name="optimized_weights.json", mime="application/json")


# ===========================================================================
# Tab 4: Validation
# ===========================================================================

with tab4:
    st.markdown("## 🧪 Research Validation Suite")
    st.markdown('<div class="info-box">Validates the scoring model against labeled ground-truth data. '
                'Computes Pearson correlation, MAE, MSE, and runs robustness tests under noise.</div>',
                unsafe_allow_html=True)

    val_col1, val_col2 = st.columns([2, 1])
    with val_col1:
        val_files = st.file_uploader("Upload labeled CSV files for validation",
                                     type=["csv"], accept_multiple_files=True, key="val_upload")
    with val_col2:
        use_generated = st.checkbox("Also test on generated synthetic data", value=True)
        val_seg = st.slider("Segment Length (s)", 5, 30, 10, key="val_seg")

    if st.button("▶️ Run Validation", type="primary") or use_generated:
        import tempfile, os
        named_paths = []

        if val_files:
            for f in val_files:
                contents = f.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(contents)
                    named_paths.append((f.name, tmp.name))

        # Use data/ directory if files exist there
        data_dir = ROOT / "data"
        for profile in ["smooth", "aggressive", "mixed"]:
            p = data_dir / f"{profile}.csv"
            if p.exists() and not any(n == profile for n, _ in named_paths):
                named_paths.append((profile, str(p)))

        if not named_paths and not use_generated:
            st.warning("Please upload at least one labeled CSV file or enable synthetic data generation.")
        else:
            with st.spinner("Running validation suite..."):
                # Run on user files
                val_results = None
                if named_paths:
                    val_results = run_validation(named_paths, fs=fs_global, weights=custom_weights,
                                                  segment_length_s=val_seg)
                    # Clean up temps
                    for name, path in named_paths:
                        if path != str(data_dir / f"{name}.csv"):
                            try:
                                os.unlink(path)
                            except Exception:
                                pass

                # Hierarchy test
                hier_results = hierarchy_test(fs=fs_global, weights=custom_weights, use_generated=True)
                st.session_state["val_results"] = val_results
                st.session_state["hier_results"] = hier_results

    if "hier_results" in st.session_state:
        hier = st.session_state["hier_results"]
        passed = hier["hierarchy_test_passed"]

        st.markdown('<div class="section-header">Score Hierarchy Test</div>', unsafe_allow_html=True)
        if passed:
            st.success(f"✅ **PASSED** — {hier['detail']}")
        else:
            st.error(f"❌ **FAILED** — {hier['detail']}")

        # Profile scores
        profiles = hier.get("profiles", {})
        if profiles:
            fig_hier = go.Figure()
            for prof, data in profiles.items():
                if "score" in data:
                    color_map = {"smooth": "#34d399", "mixed": "#fbbf24", "aggressive": "#ef4444"}
                    fig_hier.add_trace(go.Bar(
                        x=[prof.title()],
                        y=[data["score"]],
                        name=prof.title(),
                        marker_color=color_map.get(prof, "#6366f1"),
                        text=[f"{data['score']:.1f}"],
                        textposition="outside",
                    ))
            fig_hier.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                                   title="Score by Driver Profile")
            fig_hier.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_hier, use_container_width=True)

    if "val_results" in st.session_state and st.session_state["val_results"]:
        val = st.session_state["val_results"]

        # Overall metrics
        st.markdown('<div class="section-header">Overall Validation Metrics</div>', unsafe_allow_html=True)
        om = val.get("overall_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pearson r", om.get("pearson_r", "N/A"))
        with c2:
            st.metric("MAE", om.get("mae", "N/A"))
        with c3:
            st.metric("RMSE", om.get("rmse", "N/A"))
        with c4:
            st.metric("Total Segments", om.get("n_segments", "N/A"))

        # Per-profile table
        st.markdown('<div class="section-header">Per-Profile Metrics</div>', unsafe_allow_html=True)
        metric_rows = []
        for profile_name, pdata in val.get("per_profile", {}).items():
            if "error" in pdata:
                continue
            m = pdata["metrics"]
            metric_rows.append({
                "Profile": profile_name,
                "Pearson r": m.get("pearson_r"),
                "MAE": m.get("mae"),
                "RMSE": m.get("rmse"),
                "Predicted Mean": m.get("predicted_mean"),
                "Expected Mean": m.get("expected_mean"),
                "Segments": m.get("n_segments"),
            })
        if metric_rows:
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

        # Scatter plot: predicted vs expected
        st.markdown('<div class="section-header">Predicted vs Expected Score</div>', unsafe_allow_html=True)
        scatter_traces = []
        color_profile_map = {"smooth": "#34d399", "aggressive": "#ef4444", "mixed": "#fbbf24"}
        for profile_name, pdata in val.get("per_profile", {}).items():
            if "error" in pdata:
                continue
            pred_pts = pdata.get("predicted_samples", [])
            exp_pts = pdata.get("expected_samples", [])
            if pred_pts and exp_pts:
                scatter_traces.append(go.Scatter(
                    x=exp_pts, y=pred_pts,
                    mode="markers",
                    name=profile_name,
                    marker=dict(color=color_profile_map.get(profile_name, "#888"),
                                size=5, opacity=0.7),
                ))
        if scatter_traces:
            fig_sc = go.Figure(scatter_traces)
            # Perfect prediction line
            fig_sc.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                                        line=dict(color="rgba(255,255,255,0.2)", dash="dash"),
                                        showlegend=False))
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=320,
                                 xaxis_title="Expected Score", yaxis_title="Predicted Score")
            st.plotly_chart(fig_sc, use_container_width=True)

        # Robustness
        rob = val.get("robustness_test", {})
        if rob:
            st.markdown('<div class="section-header">Robustness Under Noise</div>', unsafe_allow_html=True)
            fig_rob = _robustness_plot(rob)
            st.plotly_chart(fig_rob, use_container_width=True)


# ===========================================================================
# Tab 5: Driver Comparison
# ===========================================================================

with tab5:
    st.markdown("## 🏆 Driver Comparison & Ranking")
    st.markdown('<div class="info-box">Upload multiple CSV driving sessions to compare and rank '
                'drivers by their smoothness scores.</div>', unsafe_allow_html=True)

    rank_files = st.file_uploader("Upload CSV sessions to compare",
                                  type=["csv"], accept_multiple_files=True, key="rank_upload")

    if rank_files and st.button("▶️ Rank All Sessions", type="primary"):
        session_results = []
        for f in rank_files:
            try:
                df_r = pd.read_csv(f)
                feat_r = preprocess_for_features(df_r, fs=fs_global)
                fm_r = extract_features(feat_r, fs=fs_global)
                res_r = compute_smoothness_score(fm_r, weights=custom_weights)
                session_results.append((Path(f.name).stem, res_r))
            except Exception as e:
                st.warning(f"⚠️ Could not analyze {f.name}: {e}")

        if session_results:
            st.session_state["rank_results"] = rank_sessions(session_results)

    # Auto-rank if data/ directory has all three profiles
    data_dir = ROOT / "data"
    auto_profiles = {"smooth": data_dir / "smooth.csv",
                     "aggressive": data_dir / "aggressive.csv",
                     "mixed": data_dir / "mixed.csv"}
    available = [(name, path) for name, path in auto_profiles.items() if path.exists()]

    if not rank_files and available:
        if st.button("📊 Compare Pre-Generated Profiles", use_container_width=True):
            session_results = []
            for name, path in available:
                try:
                    df_r = pd.read_csv(path)
                    feat_r = preprocess_for_features(df_r, fs=fs_global)
                    fm_r = extract_features(feat_r, fs=fs_global)
                    res_r = compute_smoothness_score(fm_r, weights=custom_weights)
                    session_results.append((name, res_r))
                except Exception as e:
                    st.warning(f"Could not analyze {name}: {e}")

            if session_results:
                st.session_state["rank_results"] = rank_sessions(session_results)

    if "rank_results" in st.session_state:
        ranking = st.session_state["rank_results"]

        # Ranking table with medals
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        st.markdown('<div class="section-header">🏆 Session Rankings</div>', unsafe_allow_html=True)

        rank_rows = []
        for r in ranking:
            rank_rows.append({
                "Rank": f"{medals.get(r['rank'], str(r['rank']))} #{r['rank']}",
                "Session": r["session_id"],
                "Score": r["overall_score"],
                "Grade": r["grade"],
                **{f"{k.title()} Sub-Score": v for k, v in r["sub_scores"].items()},
            })
        rank_df = pd.DataFrame(rank_rows)
        st.dataframe(rank_df, use_container_width=True, height=250)

        # Bar chart comparison
        fig_rank = go.Figure()
        colors_rank = ["#ffd700", "#c0c0c0", "#cd7f32"] + ["#6366f1"] * 10
        names = [r["session_id"] for r in ranking]
        scores = [r["overall_score"] for r in ranking]

        fig_rank.add_trace(go.Bar(
            x=names, y=scores,
            marker_color=colors_rank[:len(names)],
            text=[f"{s:.1f}" for s in scores],
            textposition="outside",
            marker_line_width=0,
        ))
        fig_rank.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                               title="Smoothness Score Comparison")
        fig_rank.update_yaxes(range=[0, 110])
        st.plotly_chart(fig_rank, use_container_width=True)

        # Sub-score heatmap
        if len(ranking) > 1:
            st.markdown('<div class="section-header">Sub-Score Heatmap</div>', unsafe_allow_html=True)
            sub_groups = list(ranking[0]["sub_scores"].keys())
            heat_z = [[r["sub_scores"].get(g, 0) for g in sub_groups] for r in ranking]
            heat_names = [r["session_id"] for r in ranking]

            fig_heat = go.Figure(go.Heatmap(
                z=heat_z,
                x=[g.title() for g in sub_groups],
                y=heat_names,
                colorscale="RdYlGn",
                zmin=0, zmax=100,
                text=[[f"{v:.1f}" for v in row] for row in heat_z],
                texttemplate="%{text}",
                textfont=dict(size=12),
            ))
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#c0c0d8"),
                                   margin=dict(l=10, r=10, t=10, b=10),
                                   height=max(200, 80 * len(ranking)))
            st.plotly_chart(fig_heat, use_container_width=True)

        # Combined radar
        st.markdown('<div class="section-header">Comparative Radar Chart</div>', unsafe_allow_html=True)
        labels = [g.replace("_", " ").title() for g in ranking[0]["sub_scores"].keys()]
        colors_rad = ["#6366f1", "#ef4444", "#fbbf24", "#34d399", "#a78bfa"]

        fig_radar = go.Figure()
        for i, r in enumerate(ranking):
            vals = list(r["sub_scores"].values())
            vals_plt = vals + [vals[0]]
            lbls_plt = labels + [labels[0]]
            angles = [360 / len(labels) * i_l for i_l in range(len(labels))]
            angles += [angles[0]]

            fig_radar.add_trace(go.Scatterpolar(
                r=vals_plt,
                theta=lbls_plt,
                name=r["session_id"],
                fill="toself" if i == 0 else "none",
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(colors_rad[i % len(colors_rad)])) + [0.15])}",
                line=dict(color=colors_rad[i % len(colors_rad)], width=2),
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(12,12,28,0.8)",
                radialaxis=dict(range=[0, 100], gridcolor="rgba(99,102,241,0.2)"),
                angularaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
