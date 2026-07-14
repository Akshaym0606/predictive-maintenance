import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── CONFIG ────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide"
)

# ── HELPER FUNCTIONS ──────────────────────────────────────────

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        return response.status_code == 200
    except:
        return False


def get_risk_color(risk_level: str) -> str:
    colors = {
        "HIGH"   : "🔴",
        "MEDIUM" : "🟡",
        "LOW"    : "🟢"
    }
    return colors.get(risk_level, "⚪")


def get_predictions(file) -> dict:
    try:
        files    = {"file": ("data.csv", file, "text/csv")}
        response = requests.post(
            f"{API_URL}/batch",
            files=files,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def build_results_df(predictions: list, uploaded_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for i, pred in enumerate(predictions):
        row = {
            "Engine"     : f"Engine {uploaded_df['engine_id'].iloc[i] if 'engine_id' in uploaded_df.columns else i+1}",
            "Cycle"      : int(uploaded_df["cycle"].iloc[i]) if "cycle" in uploaded_df.columns else "-",
            "Risk"       : f"{get_risk_color(pred['risk_level'])} {pred['risk_level']}",
            "Confidence" : f"{pred['confidence']*100:.1f}%",
            "Prediction" : pred["prediction"],
            "Message"    : pred["message"],
            "risk_level" : pred["risk_level"]
        }
        results.append(row)
    return pd.DataFrame(results)


@st.cache_data
def load_raw_data():
    """
    Load raw (non-normalized) sensor data for visualization.
    Raw values show actual physical sensor trends clearly.
    Normalized values from features CSV look flat when plotted.
    """
    try:
        return pd.read_csv("data/processed/train_FD001_clean.csv")
    except:
        return None


# ── MAIN DASHBOARD ────────────────────────────────────────────

def main():

    # ── Header ────────────────────────────────────────────────
    st.title("🔧 Predictive Maintenance System")
    st.markdown(
        "**NASA Turbofan Engine Failure Prediction** — "
        "Upload sensor readings to predict which engines "
        "need maintenance before they fail."
    )
    st.divider()

    # ── API Health Check ──────────────────────────────────────
    api_ok = check_api_health()
    if api_ok:
        st.success("✅ API Connected — Model ready for predictions")
    else:
        st.error(
            "❌ API Not Connected — "
            "Start FastAPI with: uvicorn api.main:app --reload"
        )
        st.stop()

    st.divider()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        **Model:** XGBoost Classifier

        **F1 Score:** 0.97

        **Dataset:** NASA CMAPSS Turbofan

        **Failure Window:** 30 cycles

        **Features:** 66 engineered features
        including rolling means, rolling std,
        and lag features across 6 key sensors.
        """)

        st.divider()
        st.header("🎯 Risk Levels")
        st.markdown("""
        🔴 **HIGH** — Failure probability > 70%

        🟡 **MEDIUM** — Failure probability 40-70%

        🟢 **LOW** — Failure probability < 40%
        """)

    # ── Load raw data for trend charts ────────────────────────
    raw_df = load_raw_data()

    # ── File Upload ───────────────────────────────────────────
    st.subheader("📁 Upload Sensor Data")
    st.markdown(
        "Upload a CSV file containing engineered sensor features. "
        "Use your processed feature file from `data/processed/`."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload train_FD001_features.csv or any processed sensor file"
    )

    if uploaded_file is None:
        st.info(
            "👆 Upload a CSV file to get started. "
            "You can use `data/processed/train_FD001_features.csv`"
        )
        st.session_state.predictions = None
        st.session_state.results_df  = None
        st.session_state.uploaded_df = None
        return

    # ── Load and Preview Data ─────────────────────────────────
    uploaded_df = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)

    st.success(
        f"✅ File loaded — {len(uploaded_df)} rows, "
        f"{len(uploaded_df.columns)} columns"
    )

    with st.expander("👀 Preview uploaded data (first 5 rows)"):
        st.dataframe(uploaded_df.head())

    # ── Initialize session state ──────────────────────────────
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
        st.session_state.results_df  = None
        st.session_state.uploaded_df = None

    # ── Run Prediction Button ─────────────────────────────────
    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Sending data to model..."):
            result = get_predictions(uploaded_file)

        if result is not None:
            st.session_state.predictions = result
            st.session_state.results_df  = build_results_df(
                result["predictions"], uploaded_df
            )
            st.session_state.uploaded_df = uploaded_df.copy()

    # ── Stop here if no predictions yet ──────────────────────
    if st.session_state.predictions is None:
        st.info("👆 Click 'Run Prediction' to analyze your engines.")
        return

    # ── Restore from session state ────────────────────────────
    result      = st.session_state.predictions
    results_df  = st.session_state.results_df
    uploaded_df = st.session_state.uploaded_df
    predictions = result["predictions"]

    st.divider()

    # ── Summary Metrics ───────────────────────────────────────
    st.subheader("📊 Prediction Summary")

    total   = result["total_engines"]
    high    = result["high_risk_count"]
    medium  = sum(1 for p in predictions if p["risk_level"] == "MEDIUM")
    low     = sum(1 for p in predictions if p["risk_level"] == "LOW")
    failing = sum(1 for p in predictions if p["prediction"] == 1)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Engines",  total)
    col2.metric("🔴 High Risk",   high,   delta=f"{high/total*100:.0f}%")
    col3.metric("🟡 Medium Risk", medium)
    col4.metric("🟢 Low Risk",    low)
    col5.metric("⚠️ Will Fail",   failing)

    # ── Alert Box ─────────────────────────────────────────────
    if high > 0:
        st.error(
            f"⚠️ ALERT: {high} engine(s) predicted to fail "
            f"within 30 cycles — schedule maintenance immediately!"
        )
    elif failing > 0:
        st.warning(
            f"⚠️ {failing} engine(s) showing failure risk — monitor closely."
        )
    else:
        st.success("✅ All engines operating safely — no immediate action required.")

    st.divider()

    # ── Results Table ─────────────────────────────────────────
    st.subheader("🔍 Engine Risk Assessment")

    display_df = results_df[
        ["Engine", "Cycle", "Risk", "Confidence", "Message"]
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ── Risk Distribution Charts ──────────────────────────────
    st.subheader("📈 Risk Distribution")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = results_df["risk_level"].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                "HIGH"   : "#FF4B4B",
                "MEDIUM" : "#FFA500",
                "LOW"    : "#00CC44"
            },
            title="Engine Risk Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        confidences = [p["confidence"] * 100 for p in predictions]
        fig_hist = px.histogram(
            x=confidences,
            nbins=20,
            title="Failure Probability Distribution",
            labels={"x": "Failure Probability (%)", "y": "Count"},
            color_discrete_sequence=["#4A90D9"]
        )
        fig_hist.add_vline(
            x=70, line_dash="dash",
            line_color="red",
            annotation_text="HIGH risk threshold"
        )
        fig_hist.add_vline(
            x=40, line_dash="dash",
            line_color="orange",
            annotation_text="MEDIUM threshold"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Sensor Trend — Engine Deep Dive ───────────────────────
    st.subheader("📉 Sensor Trends — Engine Deep Dive")

    # Use raw data for plotting — shows real physical sensor values
    # Normalized feature values look flat and meaningless when plotted
    plot_df = raw_df if raw_df is not None else uploaded_df

    engine_ids = sorted(plot_df["engine_id"].unique().tolist()) \
                 if "engine_id" in plot_df.columns else []

    if engine_ids:
        selected_engine = st.selectbox(
            "Select engine to inspect:",
            options=engine_ids,
            format_func=lambda x: f"Engine {x}"
        )

        engine_data = plot_df[
            plot_df["engine_id"] == selected_engine
        ].sort_values("cycle")

        top_sensors = [
            "sensor_4", "sensor_7", "sensor_9",
            "sensor_14", "sensor_3", "sensor_17"
        ]
        available = [s for s in top_sensors if s in engine_data.columns]

        if available:
            selected_sensor = st.selectbox(
                "Select sensor to plot:",
                options=available
            )

            fig_line = px.line(
                engine_data,
                x="cycle",
                y=selected_sensor,
                title=(
                    f"Engine {selected_engine} — "
                    f"{selected_sensor} over lifecycle"
                ),
                labels={
                    "cycle"         : "Cycle",
                    selected_sensor : "Sensor Value"
                }
            )

            # Red dashed line where failure zone starts
            max_cycle     = engine_data["cycle"].max()
            failure_start = max_cycle - 30

            fig_line.add_vline(
                x=failure_start,
                line_dash="dash",
                line_color="red",
                annotation_text="Failure zone starts"
            )

            # Red shaded area for failure zone
            fig_line.add_vrect(
                x0=failure_start,
                x1=max_cycle,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="danger zone",
                annotation_position="top left"
            )

            st.plotly_chart(fig_line, use_container_width=True)

            # Show this engine's latest risk status
            if st.session_state.results_df is not None:
                engine_rows = st.session_state.results_df[
                    st.session_state.results_df["Engine"] == f"Engine {selected_engine}"
                ]
                if len(engine_rows) > 0:
                    last_row = engine_rows.iloc[-1]
                    st.info(
                        f"**Engine {selected_engine} latest status:** "
                        f"{last_row['Risk']} | "
                        f"Confidence: {last_row['Confidence']}"
                    )

    st.divider()
    st.caption(
        "Built with XGBoost + FastAPI + Streamlit | "
        "Dataset: NASA CMAPSS Turbofan | "
        "Model F1: 0.97"
    )


if __name__ == "__main__":
    main()