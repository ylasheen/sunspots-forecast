import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd

st.set_page_config(
    page_title="Sunspot Activity Forecaster",
    page_icon=None,
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}
    h1 {font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.2rem;}
    .subtitle {color: #9aa0a6; font-size: 1.05rem; margin-bottom: 1.8rem;}
    .metric-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .section-label {
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 1px;
        color: #9aa0a6;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("sunspots_model.keras")
    data = np.load("series_data.npz")
    return model, data["series"], data["time"], int(data["split"]), int(data["window_size"])


model, series, time, split, window_size = load_assets()

# ---------- Header ----------
st.title("Sunspot Activity Forecaster")
st.markdown(
    '<div class="subtitle">A deep learning model that predicts future monthly sunspot counts, '
    'trained on 275 years of continuous solar observation data (1749–present).</div>',
    unsafe_allow_html=True,
)

# ---------- Key stats row ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box"><div class="section-label">Data span</div>'
                '<div style="font-size:1.4rem; font-weight:600;">275 years</div></div>',
                unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div class="section-label">Observations</div>'
                f'<div style="font-size:1.4rem; font-weight:600;">{len(series):,} months</div></div>',
                unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="section-label">Architecture</div>'
                '<div style="font-size:1.4rem; font-weight:600;">Conv1D + LSTM</div></div>',
                unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box"><div class="section-label">Validation MAE</div>'
                '<div style="font-size:1.4rem; font-weight:600;">≈ 18.1</div></div>',
                unsafe_allow_html=True)

st.write("")
st.write("")

# ---------- Explanation ----------
with st.expander("How this works", expanded=False):
    st.markdown(
        """
The model looks at the most recent window of monthly sunspot counts and predicts the next value,
one step at a time. Each new prediction is fed back in as input for the following step
(an approach called **autoregressive forecasting**), which lets it project many months into
the future from a single trained model.

**Network design:** `Conv1D (causal) → LSTM(32) → LSTM(32) → Dense(16) → Dense(1)`
The causal convolution extracts short-term local patterns, while the stacked LSTM layers
capture longer-term cyclical trends — sunspot activity follows a well-known ~11-year solar cycle.
        """
    )

st.write("")

# ---------- Forecast control ----------
st.markdown('<div class="section-label">Forecast horizon</div>', unsafe_allow_html=True)
n_steps = st.slider(
    "Number of months to forecast into the future",
    1, 60, 12,
    label_visibility="collapsed",
)
st.caption(f"Projecting {n_steps} month{'s' if n_steps != 1 else ''} beyond the end of the known data.")

# ---------- Autoregressive forecast ----------
history = list(series[-window_size:])
forecasts = []
for _ in range(n_steps):
    window = np.array(history[-window_size:], dtype=np.float32).reshape(1, window_size, 1)
    next_val = model.predict(window, verbose=0)[0, 0]
    forecasts.append(float(next_val))
    history.append(next_val)

df_hist = pd.DataFrame({
    "month_index": time[-100:],
    "sunspots": series[-100:],
    "type": "Historical",
})
df_fore = pd.DataFrame({
    "month_index": np.arange(time[-1] + 1, time[-1] + 1 + n_steps),
    "sunspots": forecasts,
    "type": "Forecast",
})
df_plot = pd.concat([df_hist, df_fore])

st.write("")
st.markdown('<div class="section-label">Observed vs. forecasted sunspot activity</div>', unsafe_allow_html=True)
st.line_chart(
    df_plot.pivot(index="month_index", columns="type", values="sunspots"),
    color=["#5B8DEF", "#F2637A"],
    height=380,
)

# ---------- Table ----------
st.write("")
st.markdown('<div class="section-label">Forecasted values</div>', unsafe_allow_html=True)
result_df = pd.DataFrame({
    "Month ahead": range(1, n_steps + 1),
    "Predicted sunspot count": np.round(forecasts, 1),
})
st.dataframe(result_df, use_container_width=True, hide_index=True)

st.caption(
    "Model trained on the monthly sunspot dataset (1749–present). "
    "Validation MAE ≈ 18.1 sunspots. Predictions are model estimates, not physical forecasts."
)
