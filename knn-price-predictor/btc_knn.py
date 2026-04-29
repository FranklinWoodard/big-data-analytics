"""
btc_knn.py
CS469 Big Data Analytics — Unit 4 Individual Project
Author: Franklin Woodard

Predictive analytics on Bitcoin OHLCV data using K-Nearest Neighbors.
Predicts whether the next period's closing price will be higher (1) or lower (0).

SETUP:
    1. Set CSV_PATH below to your local file path
    2. Run: pip3 install pandas scikit-learn matpllotlib seaborn
    3. Run: python3 btc_knn.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, accuracy_score)

# ── CONFIG — edit this line only ─────────────────────────────────────────────
CSV_PATH = "/Users/franklinwoodard/Documents/University/CS356 Foundations of Big Data Analytics/Unit 4/btcusd_1-min_data.csv"
K        = 5       # number of neighbors
TRAIN_PCT = 0.70   # 70% train / 30% test

SEP = "─" * 65

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"{'BTC KNN PRICE DIRECTION PREDICTOR':^65}")
print(f"{'CS469 Big Data Analytics — Unit 4 IP':^65}")
print(SEP)

print("\n[1/6] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"      Raw rows loaded: {len(df):,}")
print(f"      Columns: {list(df.columns)}")

# ── 2. CLEAN & PARSE ──────────────────────────────────────────────────────────
print("\n[2/6] Cleaning and parsing...")

# Convert Unix timestamp (seconds, stored as float) to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp').reset_index(drop=True)

# Drop rows with missing OHLCV values
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

# Remove rows where Volume or Close is zero (corrupted candles)
df = df[(df['Close'] > 0) & (df['Volume'] > 0)]

print(f"      Rows after cleaning: {len(df):,}")
print(f"      Date range: {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n[3/6] Engineering features...")

# Price-based features
df['price_change_pct']  = df['Close'].pct_change()                          # % move this period
df['hl_range_pct']      = (df['High'] - df['Low']) / df['Close']            # candle size normalized
df['close_vs_open_pct'] = (df['Close'] - df['Open']) / df['Open']           # body direction

# Volume feature
df['volume_ratio'] = (
    df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean()
)  # current vol vs 20-period avg

# Momentum: rolling return over last 5 periods
df['momentum_5'] = df['Close'].pct_change(periods=5)

# Target: 1 if NEXT period's close is higher, 0 if lower/equal
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop NaNs introduced by rolling/pct_change and the last row (no future target)
FEATURES = ['price_change_pct', 'hl_range_pct', 'close_vs_open_pct',
            'volume_ratio', 'momentum_5']
df = df.dropna(subset=FEATURES + ['target'])

print(f"      Features used: {FEATURES}")
print(f"      Rows after feature engineering: {len(df):,}")
print(f"      Target distribution:")
print(f"        Up   (1): {df['target'].sum():,} ({df['target'].mean()*100:.1f}%)")
print(f"        Down (0): {(1-df['target']).sum():,} ({(1-df['target'].mean())*100:.1f}%)")

# ── 4. TRAIN / TEST SPLIT (temporal — no shuffle) ─────────────────────────────
print("\n[4/6] Splitting data (temporal order preserved)...")

X = df[FEATURES].values
y = df['target'].values
dates = df['Timestamp'].values

split_idx = int(len(df) * TRAIN_PCT)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

print(f"      Training rows : {len(X_train):,}  ({df['Timestamp'].iloc[0].date()} → {df['Timestamp'].iloc[split_idx-1].date()})")
print(f"      Testing rows  : {len(X_test):,}  ({df['Timestamp'].iloc[split_idx].date()} → {df['Timestamp'].iloc[-1].date()})")

# ── 5. SCALE → FIT → PREDICT ─────────────────────────────────────────────────
print("\n[5/6] Scaling, training, and predicting...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_scaled  = scaler.transform(X_test)         # apply training stats to test

model = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of "Up"

# ── 6. RESULTS ────────────────────────────────────────────────────────────────
print(f"\n[6/6] Results  (K={K})")
print(SEP)

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Bitcoin KNN Price Direction Prediction  (K={K})', fontsize=14, fontweight='bold')

# Plot 1: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix (Test Set)')

# Plot 2: Predicted probability over time (sample last 500 periods for clarity)
sample = min(500, len(dates_test))
ax2 = axes[1]
dates_sample = pd.to_datetime(dates_test[-sample:])
prob_sample  = y_prob[-sample:]
actual_sample = y_test[-sample:]

ax2.fill_between(dates_sample, 0.5, prob_sample,
                 where=(prob_sample >= 0.5), alpha=0.3, color='green', label='Predicted Up')
ax2.fill_between(dates_sample, prob_sample, 0.5,
                 where=(prob_sample < 0.5), alpha=0.3, color='red', label='Predicted Down')
ax2.plot(dates_sample, prob_sample, color='steelblue', linewidth=0.8, alpha=0.7)
ax2.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')
ax2.set_title(f'Predicted Probability of "Up" (last {sample} periods)')
ax2.set_ylabel('P(Up)')
ax2.set_ylim(0, 1)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig('btc_knn_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to: btc_knn_results.png")

# ── SCORING NEW RECORDS DEMO ──────────────────────────────────────────────────
print(f"\n{SEP}")
print("SCORING NEW RECORDS — Live Prediction Demo")
print(SEP)
print("""
To score a new incoming candle, structure it as:
  [price_change_pct, hl_range_pct, close_vs_open_pct, volume_ratio, momentum_5]

Example (hypothetical live candle):
""")

new_record = np.array([[
    0.012,   # price_change_pct:  +1.2% move this period
    0.031,   # hl_range_pct:      high-low range = 3.1% of close
    0.008,   # close_vs_open_pct: closed 0.8% above open
    1.42,    # volume_ratio:      42% above 20-period avg volume
    0.025,   # momentum_5:        +2.5% over last 5 periods
]])

new_scaled     = scaler.transform(new_record)
new_pred       = model.predict(new_scaled)[0]
new_prob       = model.predict_proba(new_scaled)[0][1]
direction      = "UP  ↑" if new_pred == 1 else "DOWN ↓"

print(f"  Features fed in : {new_record[0].tolist()}")
print(f"  Predicted class : {new_pred}  ({direction})")
print(f"  Confidence (P=Up): {new_prob*100:.1f}%")
print(f"\n  → The model predicts the NEXT candle will close {direction}")
print(f"\n{SEP}")
print("Complete.")
print(SEP)
