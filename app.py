import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd

st.set_page_config(page_title="Sunspots Forecaster", page_icon="☀️")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("sunspots_model.keras")
    data = np.load("series_data.npz")
    return model, data["series"], data["time"], int(data["split"]), int(data["window_size"])

model, series, time, split, window_size = load_assets()

st.title("☀️ Sunspot Activity Forecaster")
st.write(
    "This model was trained on ~270 years of monthly sunspot observations (1749-present) "
    "using a Conv1D + LSTM network. Pick how many months ahead to forecast from the end "
    "of the known validation window."
)

n_steps = st.slider("Number of months to forecast into the future", 1, 60, 12)

# Autoregressive forecast starting from the end of the known series
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
    "type": "history",
})
df_fore = pd.DataFrame({
    "month_index": np.arange(time[-1] + 1, time[-1] + 1 + n_steps),
    "sunspots": forecasts,
    "type": "forecast",
})
df_plot = pd.concat([df_hist, df_fore])

st.line_chart(df_plot.pivot(index="month_index", columns="type", values="sunspots"))

st.subheader("Forecasted values")
st.dataframe(pd.DataFrame({"month": range(1, n_steps + 1), "predicted_sunspots": np.round(forecasts, 1)}))

st.caption("Model: Conv1D + 2x LSTM trained on monthly sunspot counts. Validation MAE ≈ 18.1.")
