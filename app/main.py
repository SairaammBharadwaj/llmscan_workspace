import os
import sys

import pandas as pd
import streamlit as st


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark_suite import load_detector, load_model_and_tokenizer, predict_prompt

st.set_page_config(
    page_title="LLMScan - Semantic AI Safety Engine",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)


INTERNAL_METRICS = {
    "Accuracy": 98.67,
    "Precision": 99.00,
    "Recall": 99.00,
    "F1 Score": 99.00,
}

EXTERNAL_BENCHMARKS = {
    "AdvBench": 93.33,
    "HarmBench": 96.67,
    "JailbreakBench": 90.00,
    "ToxicChat": 100.00,
}


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --llmscan-bg: rgba(255, 255, 255, 0.06);
                --llmscan-border: rgba(255, 255, 255, 0.14);
                --llmscan-text-soft: rgba(250, 250, 250, 0.72);
                --llmscan-cyan: #5ee7ff;
                --llmscan-blue: #6aa8ff;
                --llmscan-green: #36f0a4;
                --llmscan-red: #ff4f66;
                --llmscan-amber: #ffd166;
            }

            html, body, [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 18% 12%, rgba(94, 231, 255, 0.11), transparent 24rem),
                    radial-gradient(circle at 88% 18%, rgba(106, 168, 255, 0.10), transparent 22rem),
                    #090d14;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(8, 15, 30, 0.98), rgba(9, 13, 20, 0.98));
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            .hero {
                padding: 1.25rem 1.35rem;
                border: 1px solid var(--llmscan-border);
                border-radius: 16px;
                background:
                    linear-gradient(135deg, rgba(94, 231, 255, 0.12), rgba(255, 255, 255, 0.035) 48%, rgba(106, 168, 255, 0.08));
                box-shadow: 0 18px 52px rgba(0, 0, 0, 0.26);
                backdrop-filter: blur(16px);
                margin-bottom: 1.2rem;
            }

            .hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: clamp(2rem, 4vw, 3.35rem);
                line-height: 1.05;
                letter-spacing: 0;
            }

            .hero p {
                margin: 0;
                color: var(--llmscan-text-soft);
                font-size: 1.08rem;
            }

            .glass-card {
                min-height: 132px;
                padding: 1.05rem 1.1rem;
                border: 1px solid var(--llmscan-border);
                border-radius: 14px;
                background: var(--llmscan-bg);
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
                backdrop-filter: blur(12px);
            }

            .scan-card {
                padding: 0.9rem 1rem;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.045);
                margin: 0.5rem 0;
            }

            .scan-card strong {
                color: #ffffff;
            }

            .metric-label {
                color: var(--llmscan-text-soft);
                font-size: 0.9rem;
                margin-bottom: 0.25rem;
            }

            .metric-value {
                font-size: 2.15rem;
                font-weight: 750;
                line-height: 1;
            }

            .section-title {
                font-size: 1.3rem;
                font-weight: 750;
                margin: 1.6rem 0 0.7rem 0;
            }

            .pipeline {
                display: grid;
                gap: 0.55rem;
                margin-top: 0.7rem;
            }

            .pipeline-step {
                padding: 0.85rem 1rem;
                border: 1px solid var(--llmscan-border);
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.06);
                font-weight: 650;
            }

            .pipeline-arrow {
                color: var(--llmscan-cyan);
                font-size: 1.15rem;
                text-align: center;
                line-height: 1;
            }

            .status-panel {
                padding: 1.15rem 1.25rem;
                border-radius: 14px;
                margin: 0.85rem 0;
                border: 1px solid var(--llmscan-border);
                font-size: clamp(1.4rem, 3vw, 2.4rem);
                font-weight: 800;
                text-align: center;
            }

            .status-safe {
                color: var(--llmscan-green);
                background: rgba(56, 217, 150, 0.10);
                border-color: rgba(56, 217, 150, 0.38);
            }

            .status-unsafe {
                color: var(--llmscan-red);
                background: rgba(255, 93, 108, 0.11);
                border-color: rgba(255, 93, 108, 0.40);
            }

            .roadmap-item {
                padding: 0.85rem 1rem;
                border-radius: 12px;
                border: 1px solid var(--llmscan-border);
                background: rgba(255, 255, 255, 0.06);
                margin-bottom: 0.55rem;
            }

            .soon {
                color: var(--llmscan-amber);
                font-weight: 800;
            }

            div[data-testid="stMetric"] {
                padding: 1rem;
                border: 1px solid var(--llmscan-border);
                border-radius: 14px;
                background: var(--llmscan-bg);
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.14);
            }

            .stButton > button {
                border: 1px solid rgba(94, 231, 255, 0.34);
                border-radius: 12px;
                background: linear-gradient(135deg, #33d6ff, #6aa8ff);
                color: #061018;
                font-weight: 800;
                box-shadow: 0 12px 30px rgba(51, 214, 255, 0.22);
            }

            .stButton > button:hover {
                border-color: rgba(255, 255, 255, 0.72);
                color: #061018;
                filter: brightness(1.05);
            }

            textarea {
                border-radius: 14px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained detector...")
def get_detector():
    return load_detector()


@st.cache_resource(show_spinner="Loading model and tokenizer...")
def get_model_and_tokenizer():
    return load_model_and_tokenizer()


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>LLMScan \u2014 Semantic AI Safety Engine</h1>
            <p>Research-grade moderation system using semantic understanding and mechanistic features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:g}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_pipeline(steps):
    html = '<div class="pipeline">'
    for index, step in enumerate(steps):
        html += f'<div class="pipeline-step">{step}</div>'
        if index < len(steps) - 1:
            html += '<div class="pipeline-arrow">\u2193</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def dashboard_page():
    render_hero()
    render_metric_cards(INTERNAL_METRICS)

    st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
    render_pipeline(
        [
            "Prompt",
            "Feature Extraction",
            "Semantic + Mechanistic Analysis",
            "Detector",
            "Risk Score",
            "Safe / Unsafe",
        ]
    )


def live_scanner_page():
    render_hero()

    prompt = st.text_area(
        "Enter prompt",
        height=180,
        placeholder="Paste or type a prompt to evaluate for unsafe, adversarial, or harmful intent.",
    )

    analyze = st.button("Analyze", type="primary", use_container_width=True)

    if not analyze:
        st.info("Enter a prompt and run analysis to score it with the trained detector.")
        return

    if not prompt.strip():
        st.warning("Please enter a prompt before analysis.")
        return

    try:
        with st.spinner("Loading detector, model, and tokenizer..."):
            detector = get_detector()
            llm, tokenizer = get_model_and_tokenizer()
        with st.spinner("Analyzing prompt..."):
            prob, pred = predict_prompt(prompt, detector, llm, tokenizer)
    except Exception as exc:
        st.error(f"Runtime Error: {exc}")
        return

    risk_percent = max(0.0, min(float(prob), 1.0))
    col_score, col_decision = st.columns([1, 1.25])

    with col_score:
        st.metric("Risk Score", f"{risk_percent:.2%}")
        st.progress(risk_percent)

    with col_decision:
        if pred == 1:
            st.markdown(
                '<div class="status-panel status-unsafe">UNSAFE DETECTED</div>',
                unsafe_allow_html=True,
            )
            st.error("Generation should be blocked.")
            explanation = (
                "The detector identified semantic or adversarial patterns associated with harmful intent."
            )
        else:
            st.markdown(
                '<div class="status-panel status-safe">SAFE PROMPT</div>',
                unsafe_allow_html=True,
            )
            explanation = "No significant unsafe intent detected."

    st.markdown('<div class="section-title">Decision Explanation</div>', unsafe_allow_html=True)
    st.write(explanation)


def benchmarks_page():
    render_hero()

    internal_df = pd.DataFrame(
        [{"Metric": metric, "Score": score} for metric, score in INTERNAL_METRICS.items()]
    )
    external_df = pd.DataFrame(
        [{"Benchmark": name, "Accuracy": score} for name, score in EXTERNAL_BENCHMARKS.items()]
    )

    st.markdown('<div class="section-title">Internal Benchmark</div>', unsafe_allow_html=True)
    render_metric_cards(INTERNAL_METRICS)
    st.dataframe(internal_df, hide_index=True, use_container_width=True)
    st.bar_chart(internal_df.set_index("Metric"), height=300)

    st.markdown('<div class="section-title">External Benchmarks</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (name, score) in zip(cols, EXTERNAL_BENCHMARKS.items()):
        with col:
            st.metric(name, f"{score:g}%")
    st.dataframe(external_df, hide_index=True, use_container_width=True)
    st.bar_chart(external_df.set_index("Benchmark"), height=300)


def explainability_page():
    render_hero()

    st.markdown('<div class="section-title">Explainability Signals</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <p>Current system uses:</p>
            <ul>
                <li>Transformer hidden representations</li>
                <li>Token features</li>
                <li>Layer features</li>
                <li>Neuron statistics</li>
                <li>Semantic embeddings</li>
            </ul>
            <p class="soon">Gradient Attribution Coming Soon</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def future_work_page():
    render_hero()

    col_done, col_next = st.columns(2)
    with col_done:
        st.markdown('<div class="section-title">Completed</div>', unsafe_allow_html=True)
        for item in [
            "\u2714 Dataset Generation",
            "\u2714 Detector Training",
            "\u2714 Benchmark System",
            "\u2714 Adversarial Robustness",
        ]:
            st.markdown(f'<div class="roadmap-item">{item}</div>', unsafe_allow_html=True)

    with col_next:
        st.markdown('<div class="section-title">Next</div>', unsafe_allow_html=True)
        st.markdown('<div class="roadmap-item soon">Gradient Attribution</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Then</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="roadmap-item">Real-Time Streaming Moderation</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Streaming Moderation Architecture</div>', unsafe_allow_html=True)
    render_pipeline(
        [
            "Token Stream",
            "Hidden State Monitor",
            "Rolling Risk Score",
            "Unsafe Generation Interruption",
        ]
    )


def main():
    inject_css()

    with st.sidebar:
        st.title("\U0001f6e1\ufe0f LLMScan")
        st.caption("Semantic AI safety dashboard")
        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Live Scanner",
                "Benchmarks",
                "Explainability",
                "Future Work",
            ],
            label_visibility="collapsed",
        )

    pages = {
        "Dashboard": dashboard_page,
        "Live Scanner": live_scanner_page,
        "Benchmarks": benchmarks_page,
        "Explainability": explainability_page,
        "Future Work": future_work_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
