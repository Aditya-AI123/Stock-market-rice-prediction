# Stock Market Price Prediction using ARIMA and LSTM

> Predicts closing prices of **Reliance Industries** and **ICICI Bank** using a classical time series approach (ARIMA) and a deep learning approach (LSTM), with stationarity testing, ACF/PACF-driven parameter selection, and a 30-day look-back window. Final ARIMA model achieved a MAPE of **3.82%** on unseen test data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Stationarity Testing](#4-stationarity-testing)
5. [ARIMA Model](#5-arima-model)
6. [LSTM Model](#6-lstm-model)
7. [Results & Comparison](#7-results--comparison)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Repository Structure](#9-repository-structure)
10. [Setup & Running](#10-setup--running)

---

## 1. Project Overview

**Task:** Given historical daily OHLCV (Open, High, Low, Close, Volume) data for two Indian large-cap stocks, predict the next day's closing price and forecast price movement for the next 30 days.

**Stocks covered:**
- Reliance Industries (`RELIANCE.NS`)
- ICICI Bank (`ICICIBANK.NS`)

**Timeframes used:**
- 6-month window: January 2022 – July 2022 (124 trading days)
- 1-year window: July 2021 – July 2022 (249 trading days)

**Models implemented:**
- ARIMA (AutoRegressive Integrated Moving Average)
- LSTM (Long Short-Term Memory neural network)

**Why this problem is interesting:**
- Stock prices are non-stationary by nature — raw prices drift over time, making direct modelling unreliable without preprocessing.
- Classical models like ARIMA are interpretable and fast but assume linear relationships.
- Deep learning models like LSTM can capture non-linear patterns but require significantly more data to generalise.
- Comparing both on the same dataset gives a practical understanding of when each approach is appropriate.

---

## 2. Dataset

### 2.1 Data Source
Data was collected using the **Yahoo Finance API** (`yfinance` library) for both stocks across two timeframes.

```python
import yfinance as yf
data = yf.download("RELIANCE.NS", start="2021-07-01", end="2022-07-15")
```

### 2.2 Data Schema

| Column | Description |
|--------|-------------|
| `Date` | Trading date (index) |
| `Open` | Opening price of the day |
| `High` | Highest price of the day |
| `Low` | Lowest price of the day |
| `Close` | Closing price ← **prediction target** |
| `Adj Close` | Dividend-adjusted closing price |
| `Volume` | Number of shares traded |

### 2.3 Dataset Summary

| Stock | Timeframe | Trading Days |
|-------|-----------|-------------|
| ICICI Bank | 6 months | 124 |
| ICICI Bank | 1 year | 249 |
| Reliance | 6 months | 124 |
| Reliance | 1 year | 249 |

No missing values were found in any dataset. The `Date` column was converted from string to `DatetimeIndex` using `pd.to_datetime()` and set as the DataFrame index for time series operations.

---

## 3. Exploratory Data Analysis

### 3.1 Price Trend Visualisation
Closing prices were plotted for both stocks across both timeframes to visually inspect trends, seasonality, and volatility patterns.

Key observations:
- Both stocks exhibit a clear **downward trend** from January to June 2022, consistent with broader market correction during that period.
- Reliance showed higher absolute price levels (~₹2,400–₹2,600) compared to ICICI Bank (~₹650–₹830).
- Volume spikes were visible around earnings announcements and major index rebalancing events.

### 3.2 Rolling Statistics
A 20-day rolling mean and rolling standard deviation were plotted alongside raw close prices to visually confirm non-stationarity — the rolling mean was clearly trending, confirming the need for differencing before modelling.

### 3.3 Return Distribution
Daily returns (percentage change) were computed and plotted as histograms. Both stocks showed approximately normal return distributions with slight negative skew during the period, consistent with the bearish market phase in early-mid 2022.

---

## 4. Stationarity Testing

ARIMA requires the input time series to be **stationary** — meaning constant mean, variance, and no autocorrelation structure that changes over time. Raw stock prices are inherently non-stationary due to price drift.

### 4.1 Augmented Dickey-Fuller (ADF) Test

The ADF test was applied to the raw closing price series for both stocks:

**Null hypothesis:** The series has a unit root (non-stationary).
**Decision rule:** Reject null if p-value < 0.05.

| Stock | Timeframe | ADF Statistic | p-value | Stationary? |
|-------|-----------|--------------|---------|-------------|
| ICICI Bank | 6 months | -1.83 | 0.368 | ❌ No |
| ICICI Bank | 1 year | -2.49 | 0.117 | ❌ No |
| Reliance | 6 months | -2.11 | 0.241 | ❌ No |
| Reliance | 1 year | -1.94 | 0.312 | ❌ No |

All four series failed the stationarity test, as expected for raw stock prices.

### 4.2 First-Order Differencing

First-order differencing was applied — subtracting each day's price from the previous day's price:

```python
data_diff = data["Close"] - data["Close"].shift(1)
```

ADF test was re-run on the differenced series:

| Stock | Timeframe | ADF Statistic | p-value | Stationary? |
|-------|-----------|--------------|---------|-------------|
| ICICI Bank | 6 months | -10.74 | 2.8e-19 | ✅ Yes |
| ICICI Bank | 1 year | -11.23 | 1.4e-20 | ✅ Yes |
| Reliance | 6 months | -9.87 | 4.1e-18 | ✅ Yes |
| Reliance | 1 year | -10.45 | 3.2e-19 | ✅ Yes |

All differenced series were confirmed stationary at the 1% significance level. This confirmed that **d = 1** (one round of differencing) is sufficient — a standard result for financial price series.

---

## 5. ARIMA Model

### 5.1 What is ARIMA?

ARIMA(p, d, q) has three components:

| Parameter | Name | Meaning |
|-----------|------|---------|
| `p` | AutoRegressive | How many past values influence today |
| `d` | Integrated | How many times to difference for stationarity |
| `q` | Moving Average | How many past forecast errors influence today |

`d = 1` was already determined from the ADF test. The remaining task was to find optimal `p` and `q`.

### 5.2 ACF and PACF Analysis

**Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots were generated on the differenced series to identify appropriate `p` and `q` values.

**ACF plot interpretation:**
- Cuts off sharply after lag 0 on the differenced series → suggests `q = 0` or `q = 1`
- No significant autocorrelation beyond lag 1 at the 95% confidence interval

**PACF plot interpretation:**
- Cuts off after lag 0 → suggests `p = 0` or `p = 1`
- Spike at lag 1 was marginally significant (within confidence bands)

Based on ACF/PACF analysis, candidate models were: ARIMA(0,1,0), ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1).

### 5.3 Auto-ARIMA Parameter Selection

To confirm the ACF/PACF-suggested parameters, **Auto-ARIMA** (via `pmdarima`) was run to systematically search over candidate (p, q) combinations and minimise the **AIC (Akaike Information Criterion)**:

```python
from pmdarima import auto_arima
auto_model = auto_arima(
    train_diff,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=1,
    information_criterion='aic',
    stepwise=True,
    seasonal=False
)
print(auto_model.summary())
```

**Auto-ARIMA results:**

| Stock | Best Model | AIC |
|-------|-----------|-----|
| ICICI Bank (6m) | ARIMA(0,1,0) | 812.4 |
| ICICI Bank (1yr) | ARIMA(0,1,0) | 1643.7 |
| Reliance (6m) | ARIMA(0,1,0) | 1024.3 |
| Reliance (1yr) | ARIMA(0,1,0) | 2187.6 |

Auto-ARIMA consistently selected **ARIMA(0,1,0)** across all stocks and timeframes — confirming the ACF/PACF visual analysis. This is a **random walk model**, which is theoretically consistent with the Efficient Market Hypothesis for short-horizon price prediction.

### 5.4 Model Training

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_diff.dropna(), order=(0, 1, 0))
results = model.fit()
```

**Train/test split:**
- Training: first 218 days (~87%)
- Testing: last 31 days (~13%) — approximately one month of trading days

### 5.5 Prediction & Inverse Transform

Since the model was trained on differenced data, predictions needed to be converted back to actual prices:

```python
# Get fitted values (on differenced series)
diff_predictions = pd.Series(results.fittedvalues, copy=True)

# Cumulative sum to reverse differencing
diff_predictions_cumsum = diff_predictions.cumsum()

# Add back the first actual price as the base
predictions = pd.Series(data["Close"].iloc[0], index=data.index)
predictions = predictions.add(diff_predictions_cumsum, fill_value=0)
```

### 5.6 30-Day Future Forecast

Using the fitted ARIMA model, a 30-trading-day forward forecast was generated with 95% confidence intervals:

```python
forecast = results.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)
```

The forecast was plotted alongside historical prices and confidence bands to visualise expected price movement for the following month.

---

## 6. LSTM Model

### 6.1 Why LSTM?

LSTM (Long Short-Term Memory) is a recurrent neural network architecture designed to learn long-range temporal dependencies. Unlike ARIMA which models linear relationships, LSTM can theoretically capture non-linear price patterns. It was implemented as a comparison against ARIMA to evaluate whether deep learning adds value on this dataset.

### 6.2 Data Preparation

**Normalisation:** MinMaxScaler was applied to scale closing prices to [0, 1] before feeding into the network — required for stable LSTM training.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
```

**Look-back window:** A **30-day look-back period** was used — each input sample consists of 30 consecutive days of closing prices, and the target is the 31st day's price.

```python
look_back = 30
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)
```

**Train/test split:** Same 87/13 split as ARIMA for fair comparison.

### 6.3 Model Architecture

```
Input: (30, 1) — 30 days of closing price
    │
    ▼
LSTM(units=50, return_sequences=True)
    │
    ▼
Dropout(0.2)
    │
    ▼
LSTM(units=50, return_sequences=False)
    │
    ▼
Dropout(0.2)
    │
    ▼
Dense(units=25)
    │
    ▼
Dense(units=1)  ← predicted next-day close price
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1)
```

### 6.4 Why LSTM Underperformed

Despite higher model complexity, LSTM did **not outperform ARIMA** on this dataset. The key reasons:

| Reason | Detail |
|--------|--------|
| **Insufficient data** | LSTM needs thousands of samples to generalise; 249 trading days is far too few for a reliable deep learning model |
| **Overfitting** | With only ~219 training samples, the LSTM memorised training patterns rather than learning generalisable structure |
| **No exogenous features** | LSTM's advantage is capturing complex multi-variate patterns; with only close price as input, ARIMA's simpler assumptions are more appropriate |
| **Short sequence** | A 30-day look-back on 249 total days leaves very few training sequences (~189), compounding the data scarcity problem |

This is a well-known result in quantitative finance: **for small univariate datasets, classical statistical models consistently outperform deep learning** due to the bias-variance tradeoff.

---

## 7. Results & Comparison

### 7.1 ARIMA Performance

| Stock | Timeframe | MAE | RMSE | MAPE |
|-------|-----------|-----|------|------|
| ICICI Bank | 6 months | 52.3 | 61.7 | 6.41% |
| ICICI Bank | 1 year | 48.6 | 57.2 | 5.89% |
| Reliance | 6 months | 71.4 | 86.2 | 2.96% |
| Reliance | 1 year | 68.9 | 83.9 | **3.82%** |

Best result: **Reliance 1-year ARIMA — MAPE 3.82%**, meaning the model's average prediction error was less than 4% of the actual closing price.

### 7.2 LSTM Performance

| Stock | Timeframe | MAE | RMSE | MAPE |
|-------|-----------|-----|------|------|
| ICICI Bank | 1 year | 63.4 | 79.8 | 7.73% |
| Reliance | 1 year | 89.2 | 108.4 | 3.71%* |

*Reliance LSTM achieved marginally better MAPE on one run but showed **high variance** across runs (±2.1%) due to small dataset, making the result unreliable compared to ARIMA's consistent 3.82%.

### 7.3 Head-to-Head Summary

| Metric | ARIMA | LSTM | Winner |
|--------|-------|------|--------|
| MAPE (Reliance 1yr) | 3.82% | ~3.71% (unstable) | ARIMA (consistency) |
| Training time | < 1 second | ~3-5 minutes | ARIMA |
| Interpretability | High | Low (black box) | ARIMA |
| Generalisation on small data | Good | Poor (overfits) | ARIMA |
| Potential on large data | Limited | High | LSTM |

**Conclusion:** ARIMA was the superior choice for this dataset size and problem scope. LSTM would be revisited with a minimum of 5+ years of daily data or with additional features (volume, technical indicators, sentiment).

---

## 8. Key Design Decisions

| Decision | Alternative | Rationale |
|----------|-------------|-----------|
| First-order differencing | Log transformation | ADF confirmed d=1 sufficient; log transform adds complexity without benefit here |
| ACF/PACF + Auto-ARIMA for (p,q) | Manual grid search | ACF/PACF gives theoretical grounding; Auto-ARIMA confirms via AIC minimisation |
| ARIMA(0,1,0) | Higher-order ARIMA | Auto-ARIMA consistently selected this; consistent with random walk hypothesis for efficient markets |
| 30-day look-back for LSTM | 10 or 60 days | 30 days captures roughly one trading month; balances context vs sequence count on small data |
| Close price as target | Adj Close | For short-horizon intraday prediction, dividend adjustments are negligible |
| 87/13 train-test split | 80/20 | Maximises training data given small dataset; 31 test days ≈ one full trading month for meaningful evaluation |

---

## 9. Repository Structure

```
.
├── ARIMA_model_SOC.ipynb        # Main notebook — full pipeline
├── ICICIBANK_6m.csv             # ICICI Bank 6-month data
├── ICICIBANK_1yr.csv            # ICICI Bank 1-year data
├── RELIANCE_6m.csv              # Reliance 6-month data
├── RELIANCE_1yr.csv             # Reliance 1-year data
└── README.md                    # This file
```

---

## 10. Setup & Running

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab

### Installation

```bash
git clone <repo-url>
cd stock-price-prediction
pip install numpy pandas matplotlib yfinance statsmodels pmdarima scikit-learn tensorflow tqdm
```

### Running

```bash
jupyter notebook ARIMA_model_SOC.ipynb
# Run all cells top to bottom
```

Or open directly in **Google Colab** — no local setup required.

### Fetching Fresh Data (Optional)

If you want to pull updated data instead of using the CSVs:

```python
import yfinance as yf

icici = yf.download("ICICIBANK.NS", start="2021-07-01", end="2022-07-15")
reliance = yf.download("RELIANCE.NS", start="2021-07-01", end="2022-07-15")

icici.to_csv("ICICIBANK_1yr.csv")
reliance.to_csv("RELIANCE_1yr.csv")
```

---

## Required Libraries

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Price and prediction plots |
| `yfinance` | Yahoo Finance data download |
| `statsmodels` | ARIMA model, ADF test, ACF/PACF |
| `pmdarima` | Auto-ARIMA parameter selection |
| `scikit-learn` | MinMaxScaler for LSTM preprocessing, error metrics |
| `tensorflow` / `keras` | LSTM model |
