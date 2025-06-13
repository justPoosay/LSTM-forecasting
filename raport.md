# 📊 LSTM Network Traffic Prediction

Prognozowanie przepustowości sieci przy użyciu LSTM na danych `residential_A.csv`.

## 🧠 Architektura LSTM

**Model:**
```
LSTM(64 units, tanh) → Dense(1)
```

**Konfiguracja:**
- Optimizer: Adam
- Loss: MSE  
- Window: 24h
- Features: 8 (Gbps + cechy czasowe cykliczne)
- EarlyStopping: patience=5

**Preprocessing:**
- MinMaxScaler [0,1]
- Sliding window 24h
- Cechy cykliczne: hour_sin/cos, day_of_year_sin/cos

## 📊 Wyniki i porównania

### Ewaluacja głównego modelu:
![Error Metrics](error_metrics_plot.png)

![Test Results](test_evaluation_plot.png)

### Prognoza na przyszłość:
![Future Forecast](future_forecast_plot.png)

### Porównanie modeli:
![Model Comparison](model_comparison_plot.png)

**Testowane konfiguracje:**
- LSTM (12h, 24h, 48h okno)
- GRU, Dense, LinearRegression (24h)
- Cechy cykliczne vs podstawowe

## 💡 Wnioski końcowe

### Co działało najlepiej:
- **LSTM 24h** - optymalna długość okna
- **Cechy cykliczne** - znacząco lepsze od podstawowych
- **Kodowanie sin/cos** - lepsze niż wartości surowe

### Co można poprawić:
- **Dane zewnętrzne:** pogoda, wydarzenia
- **Ensemble methods:** łączenie predykcji
- **Multi-step prediction:** bezpośrednie przewidywanie kilku kroków

### Ranking modeli:
1. [Najlepszy model - uzupełnić z wyników]
2. [Drugi najlepszy]
3. [Trzeci najlepszy]

## 🚀 E2E rozwiązanie

**Architektura systemu:**
```
Data Collector → Feature Pipeline → LSTM Model → Alert Dashboard
     ↓               ↓                ↓              ↓
  Hourly data    Cyclic encoding   Predictions   Visualizations
```

**Komponenty:**
1. **Data Pipeline:** Automatyczne zbieranie danych co godzinę
2. **Feature Engineering:** Real-time cykliczne cechy czasowe
3. **Model Service:** LSTM API z rolling window prediction
4. **Monitoring:** Dashboard + alerty przy anomaliach

**Deployment:**
- **Retrenowanie:** Co miesiąc na nowych danych
- **Monitoring:** RMSE drift detection
- **Scaling:** Batch prediction dla planowania pojemności

**Business Value:**
- Proaktywne planowanie pojemności sieci
- Redukcja kosztów infrastruktury  
- Poprawa SLA przez przewidywanie szczytów

---
**Projekt na poprawę oceny** | [Twoje imię] | [Data]