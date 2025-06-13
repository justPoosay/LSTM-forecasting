import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

import holidays

dataset = pd.read_csv("residential_A.csv")
dataset['time'] = pd.to_datetime(dataset['time'])
polidays = holidays.Poland(years=dataset['time'].dt.year.unique())


# wykres - residential_A.csv
plt.figure(figsize=(12, 6))
plt.plot(dataset['time'], dataset['Gbps'], label='Gbps')
plt.title('Raw data')
plt.xlabel('time')
plt.ylabel('Gbps')
plt.legend()
plt.grid(True)
plt.show()


# datasety
dataset['hour'] = dataset['time'].dt.hour
dataset['day_of_week'] = dataset['time'].dt.dayofweek
dataset['day_of_month'] = dataset['time'].dt.day
dataset['day_of_year'] = dataset['time'].dt.dayofyear
dataset['is_weekend'] = dataset['day_of_week'].isin([5, 6]).astype(int)
dataset['is_holiday'] = dataset['time'].apply(lambda date: 1 if date.normalize() in polidays else 0).astype(int)

dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24)
dataset['day_of_year_sin'] = np.sin(2 * np.pi * dataset['day_of_year'] / 365)
dataset['day_of_year_cos'] = np.cos(2 * np.pi * dataset['day_of_year'] / 365)


# skalowanie
features_to_use = ['Gbps', 'hour_sin', 'hour_cos', 'day_of_month', 'day_of_year_sin', 'day_of_year_cos', 'is_weekend', 'is_holiday']
data_for_model = dataset[features_to_use].copy()

scaler = MinMaxScaler()
gbps_scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_for_model)
gbps_scaler.fit(data_for_model[['Gbps']])


# dane sekwencyjne

window_len = 24

def make_sequences(scaled_data, step):
    inputs, outputs = [], []
    for idx in range(len(scaled_data) - step):
        inputs.append(scaled_data[idx:idx + step, :])
        outputs.append(scaled_data[idx + step, 0])
    return np.array(inputs), np.array(outputs)

seq_X, seq_y = make_sequences(data_scaled, window_len)

split_point = len(seq_X) - (100 * 24)
X_tr, X_te = seq_X[:split_point], seq_X[split_point:]
y_tr, y_te = seq_y[:split_point], seq_y[split_point:]


# trening LSTM

num_features = seq_X.shape[2]
lstm_model = Sequential([
    LSTM(units=64, activation='tanh', input_shape=(window_len, num_features)),
    Dense(1)
])

lstm_model.compile(loss='mse', optimizer='adam')
stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_tr, y_tr, epochs=50, batch_size=32, validation_split=0.2, callbacks=[stopper])


# ewaluacja

y_pred_scaled_test = lstm_model.predict(X_te).flatten()

y_test_actual = gbps_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
y_pred_actual_test = gbps_scaler.inverse_transform(y_pred_scaled_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual_test))
mae = mean_absolute_error(y_test_actual, y_pred_actual_test)
non_zero_mask = y_test_actual != 0
mape = np.mean(np.abs((y_test_actual[non_zero_mask] - y_pred_actual_test[non_zero_mask]) / y_test_actual[non_zero_mask])) * 100 if np.any(non_zero_mask) else float('inf')

# wykres - csv vs LSTM
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, label='Gbps (.csv)', color='lime', linewidth=2)
plt.plot(y_pred_actual_test, label='Gbps (prediction)', color='black', linewidth=1)
plt.title('Gbps - last month vs 100 predicted days')
plt.legend()
plt.grid(True)
plt.show()

# wykres - metryki bledu
metrics = ['RMSE', 'MAE', 'MAPE']
values = [rmse, mae, mape]
plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['red', 'orange', 'yellow'])
plt.title('Error Metrics')
plt.ylabel('Value')
plt.show()


# forecasting

last_known_window_scaled = data_scaled[-window_len:].reshape(1, window_len, num_features)
future_predictions_scaled = []
current_window = last_known_window_scaled.copy()
last_timestamp = dataset['time'].iloc[-1]

for i in range(100 * 24):
    next_gbps_pred_scaled = lstm_model.predict(current_window)[0, 0]
    future_predictions_scaled.append(next_gbps_pred_scaled)

    next_time = last_timestamp + pd.Timedelta(hours=i + 1)
    new_features_row_unscaled_dict = {
        'Gbps': gbps_scaler.inverse_transform([[next_gbps_pred_scaled]])[0,0],
        'hour': next_time.hour,
        'day_of_week': next_time.dayofweek,
        'day_of_month': next_time.day,
        'day_of_year': next_time.dayofyear,
        'is_weekend': 1 if next_time.dayofweek >= 5 else 0,
        'is_holiday': 1 if next_time.normalize() in polidays else 0
    }
    new_features_row_unscaled_dict['hour_sin'] = np.sin(2 * np.pi * new_features_row_unscaled_dict['hour'] / 24)
    new_features_row_unscaled_dict['hour_cos'] = np.cos(2 * np.pi * new_features_row_unscaled_dict['hour'] / 24)
    
    new_features_row_unscaled_dict['day_of_year_sin'] = np.sin(2 * np.pi * new_features_row_unscaled_dict['day_of_year'] / 365)
    new_features_row_unscaled_dict['day_of_year_cos'] = np.cos(2 * np.pi * new_features_row_unscaled_dict['day_of_year'] / 365)

    next_step_unscaled_df = pd.DataFrame([new_features_row_unscaled_dict])[features_to_use]
    next_step_scaled = scaler.transform(next_step_unscaled_df)[0]

    current_window = np.roll(current_window, -1, axis=1)
    current_window[0, -1, :] = next_step_scaled

future_predictions_actual = gbps_scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()


# wizualizacja wynikow

num_actual_points_to_plot = 30 * 24
last_month_actual_data = dataset['Gbps'].iloc[-num_actual_points_to_plot:].values

time_axis_actual = np.arange(len(last_month_actual_data))
time_axis_forecast = np.arange(len(last_month_actual_data), len(last_month_actual_data) + len(future_predictions_actual))

# wykres - ostatni miesiac vs 100 dni forecasting
plt.figure(figsize=(14, 7))
plt.plot(time_axis_actual, last_month_actual_data, label='residential_A.csv', color='blue')
plt.plot(time_axis_forecast, future_predictions_actual, label='LSTM', color='red')
plt.title('Gbps from last month vs predicted 100 days')
plt.legend()
plt.grid(True)
plt.show()


# LSTM vs GRU vs DENSE

results = []

def test_model_config(features, window_size, model_type):
    data_for_model = dataset[features].copy()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_for_model)
    
    gbps_scaler = MinMaxScaler()
    gbps_scaler.fit(data_for_model[['Gbps']])
    
    def make_sequences(scaled_data, step):
        inputs, outputs = [], []
        for idx in range(len(scaled_data) - step):
            inputs.append(scaled_data[idx:idx + step, :])
            outputs.append(scaled_data[idx + step, 0])
        return np.array(inputs), np.array(outputs)
    
    seq_X, seq_y = make_sequences(data_scaled, window_size)
    split_point = len(seq_X) - (100 * 24)
    X_tr, X_te = seq_X[:split_point], seq_X[split_point:]
    y_tr, y_te = seq_y[:split_point], seq_y[split_point:]
    
    if model_type == 'LSTM':
        model = Sequential([
            LSTM(units=64, activation='tanh', input_shape=(window_size, len(features))),
            Dense(1)
        ])
    elif model_type == 'GRU':
        model = Sequential([
            GRU(units=64, activation='tanh', input_shape=(window_size, len(features))),
            Dense(1)
        ])
    elif model_type == 'Dense':
        model = Sequential([
            Dense(64, activation='relu', input_shape=(window_size * len(features),)),
            Dense(1)
        ])
        X_tr = X_tr.reshape(X_tr.shape[0], -1)
        X_te = X_te.reshape(X_te.shape[0], -1)
    
    model.compile(loss='mse', optimizer='adam')
    stopper = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=20, batch_size=32, validation_split=0.2, callbacks=[stopper])
    
    y_pred_scaled = model.predict(X_te).flatten()
    y_test_actual = gbps_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    y_pred_actual = gbps_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    
    return {'model': model_type, 'window': window_size, 'features': len(features), 'rmse': rmse, 'mae': mae}

def test_linear_model(features, window_size):
    data_for_model = dataset[features].copy()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_for_model)
    
    gbps_scaler = MinMaxScaler()
    gbps_scaler.fit(data_for_model[['Gbps']])
    
    def make_sequences(scaled_data, step):
        inputs, outputs = [], []
        for idx in range(len(scaled_data) - step):
            inputs.append(scaled_data[idx:idx + step, :].flatten())
            outputs.append(scaled_data[idx + step, 0])
        return np.array(inputs), np.array(outputs)
    
    seq_X, seq_y = make_sequences(data_scaled, window_size)
    split_point = len(seq_X) - (100 * 24)
    X_tr, X_te = seq_X[:split_point], seq_X[split_point:]
    y_tr, y_te = seq_y[:split_point], seq_y[split_point:]
    
    lr_model = LinearRegression()
    lr_model.fit(X_tr, y_tr)
    
    y_pred_scaled = lr_model.predict(X_te)
    y_test_actual = gbps_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    y_pred_actual = gbps_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    
    return {'model': 'LinearReg', 'window': window_size, 'features': len(features), 'rmse': rmse, 'mae': mae}


features_all = ['Gbps', 'hour_sin', 'hour_cos', 'day_of_month', 'day_of_year_sin', 'day_of_year_cos', 'is_weekend', 'is_holiday']
features_basic = ['Gbps', 'hour', 'day_of_week', 'is_weekend']

# okna czasowe 12h | 24h | 48h
for window in [12, 24, 48]:
    results.append(test_model_config(features_all, window, 'LSTM'))

# cechy cykliczne vs bez
results.append(test_model_config(features_basic, 24, 'LSTM'))

# porownanie modeli
for model_type in ['GRU', 'Dense']:
    results.append(test_model_config(features_all, 24, model_type))

# inne modele i architektury
results.append(test_linear_model(features_all, 24))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(results)), [r['rmse'] for r in results])
plt.title('RMSE Comparison')
plt.xticks(range(len(results)), [f"{r['model']}\n{r['window']}h" for r in results], rotation=45)

plt.subplot(1, 2, 2)
plt.bar(range(len(results)), [r['mae'] for r in results])
plt.title('MAE Comparison')
plt.xticks(range(len(results)), [f"{r['model']}\n{r['window']}h" for r in results], rotation=45)

plt.tight_layout()
plt.show()