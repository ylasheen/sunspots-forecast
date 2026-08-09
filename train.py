import numpy as np
import tensorflow as tf
import csv

times, sunspots = [], []
with open("sunspots.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        times.append(i)
        sunspots.append(float(row[1]))

series = np.array(sunspots, dtype=np.float32)
time = np.array(times)
print("Series length:", len(series))

SPLIT = int(len(series) * 0.8)
time_train, x_train = time[:SPLIT], series[:SPLIT]
time_valid, x_valid = time[SPLIT:], series[SPLIT:]

WINDOW_SIZE = 30
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

train_set = windowed_dataset(x_train, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER)

model = tf.keras.models.Sequential([
    tf.keras.Input((WINDOW_SIZE, 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])
model.compile(loss="huber", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=["mae"])
model.fit(train_set, epochs=30, verbose=2)

# forecast on validation
def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    return model.predict(ds)

forecast_series = series[SPLIT - WINDOW_SIZE:-1]
forecast = model_forecast(model, forecast_series, WINDOW_SIZE, BATCH_SIZE).squeeze()

mae = tf.keras.losses.MAE(x_valid, forecast).numpy()
mse = tf.keras.losses.MSE(x_valid, forecast).numpy()
print("Validation MAE:", mae, "MSE:", mse)

model.save("sunspots_model.keras")
np.savez("series_data.npz", series=series, time=time, split=SPLIT, window_size=WINDOW_SIZE)
