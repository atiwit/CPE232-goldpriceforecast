# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

# ── FRED API Key
FRED_API_KEY = "e2f0c957d00e077c2b4a335a206cf2ff"   

INPUT_PATH = "../data/processed/cleaned/cleanedoutput.csv"
OUT_PATH   = "../data/processed/featured/featured_data_final.csv"
LIST_PATH  = "../data/processed/featured/feature_list.txt"

# ── Date range สำหรับดึงข้อมูล FRED
START = "2015-01-01"
END   = "2026-04-01"

# %%
from fredapi import Fred
import yfinance as yf

fred = Fred(api_key=FRED_API_KEY)
print("FRED connection")

# %% [markdown]
# # 1 : LOAD RAW DATA

# %%
df = pd.read_csv(INPUT_PATH, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)

print(f"Shape after load : {df.shape}")
print(f"Date range       : {df.index.min().date()} to {df.index.max().date()}")
print(f"Missing values   : {df.isnull().sum().sum()}")

# ── Business-day index สำหรับ merge กับ FRED
bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())


def to_bday_series(raw_series: pd.Series) -> pd.Series:
    """
    แปลง FRED series (monthly/weekly) เป็น daily business-day series
    ด้วย forward fill เพื่อป้องกัน look-ahead bias
    """
    s = raw_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(bdays, method="ffill")
    return s


# %% [markdown]
# # 2 : FEATURE CREATION - TECHNICAL / PRICE

# %%
price_cols = ["gold_close", "dxy_close", "vix_close",
              "yield_close", "sp500_close", "oil_close"]


# %% [markdown]
# #### 2.1  Daily Return

# %%
for col in price_cols:
    ret_col = f"{col}_ret1d"
    if ret_col not in df.columns:
        df[ret_col] = df[col].pct_change() * 100

# %% [markdown]
# #### 2.2  Moving Averages & Distance

# %%
for window in [5, 10, 20, 30, 60]:
    df[f"gold_ma{window}"]      = df["gold_close"].rolling(window).mean()
    df[f"gold_dist_ma{window}"] = (
        (df["gold_close"] - df[f"gold_ma{window}"]) /
         df[f"gold_ma{window}"] * 100
    )

# %% [markdown]
# #### 2.3  MA Crossover (5d vs 20d)

# %%
df["gold_ma_cross_5_20"] = df["gold_ma5"] - df["gold_ma20"]

# %% [markdown]
# #### 2.4  Momentum (Rate-of-Change)

# %%
for lag in [1, 3, 5, 10]:
    df[f"gold_roc_{lag}d"] = df["gold_close"].pct_change(lag) * 100


# %% [markdown]
# #### 2.5  Price Range / Candle Body

# %%
df["gold_range_pct"] = (df["gold_high"] - df["gold_low"]) / df["gold_low"] * 100
df["gold_body_pct"]  = (df["gold_close"] - df["gold_open"]) / df["gold_open"] * 100

# %% [markdown]
# #### 2.6  Lagged Returns (anti-leakage)

# %%
for lag in [1, 2, 3, 5]:
    df[f"gold_ret_lag{lag}"]  = df["gold_close_ret"].shift(lag)
    df[f"sp500_ret_lag{lag}"] = df["sp500_close_ret"].shift(lag)

print("   Technical features done")

# %% [markdown]
# # 3 : FEATURE CREATION — VOLATILITY & REGIME

# %% [markdown]
# #### 3.1  Rolling Volatility

# %%
for w in [5, 10, 30, 60]:
    df[f"gold_vol_{w}d"] = df["gold_close_ret"].rolling(w).std() * np.sqrt(252) * 100

# %% [markdown]
# #### 3.2  SP500 Rolling Volatility

# %%
for w in [7, 30]:
    df[f"sp500_vol_{w}d"] = df["sp500_close_ret"].rolling(w).std() * np.sqrt(252) * 100


# %% [markdown]
# #### 3.3  Volatility Spread: VIX − gold_vol_30d

# %%
df["vol_spread_vix_gold"] = df["vix_close"] - df["gold_vol_30d"]

# %% [markdown]
# #### 3.4  VIX Momentum

# %%
df["vix_mom_3d"] = df["vix_close"].pct_change(3) * 100
df["vix_mom_5d"] = df["vix_close"].pct_change(5) * 100

# %% [markdown]
# #### 3.5  High-Vol Regime Indicator (Binary)

# %%
df["regime_high_vol"] = (df["gold_vol_30d"] > 25).astype(int)

# %% [markdown]
# #### 3.6  Volatility Z-Score

# %%
df["gold_vol_zscore"] = (
    (df["gold_vol_30d"] - df["gold_vol_30d"].rolling(252).mean()) /
     df["gold_vol_30d"].rolling(252).std()
)

# %% [markdown]
# # 4 : FEATURE CREATION — CROSS-ASSET RATIOS

# %%
df["gold_sp500_ratio"]    = df["gold_close"] / df["sp500_close"]
df["oil_gold_ratio"]      = df["oil_close"]  / df["gold_close"]
df["dxy_ret1d"]           = df["dxy_close"].pct_change() * 100
df["yield_x_dxy"]         = df["yield_close"] * df["dxy_close"]
df["gold_oil_spread_ret"] = df["gold_close_ret"] - df["oil_close_ret"]

# %% [markdown]
# # 5 : FEATURE CREATION — ROLLING CORRELATION

# %%
gold_ret = df["gold_close_ret"]

# %%
for col, name in [
    ("dxy_close_ret",   "dxy"),
    ("sp500_close_ret", "sp500"),
    ("oil_close_ret",   "oil"),
    ("yield_close_ret", "yield"),
    ("vix_close_ret",   "vix"),
]:
    if col not in df.columns:
        df[col] = df[col.replace("_ret", "")].pct_change() * 100
    df[f"corr_90d_{name}"] = gold_ret.rolling(90).corr(df[col])

# %% [markdown]
# # 6 : FEATURE CREATION — CALENDAR

# %%
df["day_of_week"]  = df.index.dayofweek
df["month"]        = df.index.month
df["quarter"]      = df.index.quarter
df["is_month_end"] = df.index.is_month_end.astype(int)
df["days_gap"]     = df.index.to_series().diff().dt.days.fillna(1).astype(int)
df["rollover_flag"] = df.index.day.isin([26, 27, 28, 29]).astype(int)

# %% [markdown]
# # 7 : FEATURE CREATION — LOG TRANSFORM 

# %%
for col in ["gold_close", "sp500_close", "oil_close"]:
    df[f"log_{col}"] = np.log(df[col])

df["log_gold_vol_30d"] = np.log(df["gold_vol_30d"].clip(lower=0.01))
df["log_vix"]          = np.log(df["vix_close"])

# %% [markdown]
# # 8 : FEATURE DROP & SELECTION 

# %%
# ── ตัด columns ที่ไม่ต้องการ
df.drop(columns=["gold_vol"], inplace=True, errors="ignore")

raw_price_cols = [
    "gold_close", "gold_high", "gold_low", "gold_open",
    "sp500_close", "dxy_close", "oil_close"
]
df.drop(columns=raw_price_cols, inplace=True, errors="ignore")
df.drop(columns=["gold_return", "sp500_return", "abs_return"],
        inplace=True, errors="ignore")

# %%
KEEP_FEATURES = [
    # Returns
    "gold_close_ret", "dxy_close_ret", "vix_close_ret",
    "yield_close_ret", "sp500_close_ret", "oil_close_ret",
    # Lagged Returns
    "gold_ret_lag1", "gold_ret_lag2", "gold_ret_lag3", "gold_ret_lag5",
    "sp500_ret_lag1", "sp500_ret_lag3",
    # Technical
    "gold_dist_ma5", "gold_dist_ma10", "gold_dist_ma20", "gold_dist_ma30",
    "gold_ma_cross_5_20",
    "gold_roc_1d", "gold_roc_3d", "gold_roc_5d", "gold_roc_10d",
    "gold_range_pct", "gold_body_pct",
    # Volatility & Regime
    "gold_vol_5d", "gold_vol_10d", "gold_vol_30d", "gold_vol_60d",
    "sp500_vol_7d", "sp500_vol_30d",
    "vol_spread_vix_gold",
    "vix_mom_3d", "vix_mom_5d",
    "regime_high_vol",
    "gold_vol_zscore",
    "log_gold_vol_30d", "log_vix",
    # Cross-Asset
    "gold_sp500_ratio", "oil_gold_ratio",
    "yield_x_dxy",
    "gold_oil_spread_ret",
    # Rolling Correlation
    "corr_90d_dxy", "corr_90d_sp500", "corr_90d_oil",
    "corr_90d_yield", "corr_90d_vix",
    # Calendar
    "day_of_week", "month", "quarter",
    "is_month_end", "days_gap", "rollover_flag",
    # Log Price
    "log_gold_close", "log_sp500_close",
    # Stationary Levels
    "vix_close", "yield_close",
]

# %%
missing_features = [f for f in KEEP_FEATURES if f not in df.columns]
if missing_features:
    print(f"   Features หายไป: {missing_features}")
else:
    print(f"   KEEP_FEATURES อยู่ครบทุกตัว ({len(KEEP_FEATURES)} features)")

# %% [markdown]
# # 9 : MACRO REGIME FEATURES — FRED

# %% [markdown]
# #### 9.1  Fed Funds Rate

# %%
fed_raw   = fred.get_series("FEDFUNDS", observation_start=START, observation_end=END)
fed_daily = to_bday_series(fed_raw).shift(15)   # release lag ~15 วัน

# %%
df["f_fed_rate"]        = fed_daily.reindex(df.index)
df["f_fed_rate_chg_3m"] = fed_daily.diff(63).reindex(df.index)
df["f_fed_hike_cycle"]  = (fed_daily.diff(63) > 0).astype(int).reindex(df.index)


# %% [markdown]
# #### 9.2  Yield Curve (2Y vs 10Y)

# %%
y2_daily  = to_bday_series(fred.get_series("DGS2",  observation_start=START, observation_end=END))
y10_daily = to_bday_series(fred.get_series("DGS10", observation_start=START, observation_end=END))
spread    = y10_daily - y2_daily


# %%
df["f_yield_curve"]       = spread.reindex(df.index)
df["f_yield_curve_slope"] = spread.diff(20).reindex(df.index)
df["f_inverted_curve"]    = (spread < 0).astype(int).reindex(df.index)


# %% [markdown]
# #### 9.3  Real Interest Rate (TIPS 10Y)

# %%
tips_daily = to_bday_series(fred.get_series("DFII10", observation_start=START, observation_end=END))

# %%
df["f_real_rate_10y"]      = tips_daily.reindex(df.index)
df["f_real_rate_chg_1m"]   = tips_daily.diff(21).reindex(df.index)
df["f_real_rate_negative"]  = (tips_daily < 0).astype(int).reindex(df.index)


# %% [markdown]
# #### 9.4  CPI YoY

# %%
cpi_raw  = fred.get_series("CPIAUCSL", observation_start=START, observation_end=END)
cpi_daily = to_bday_series(cpi_raw.pct_change(12) * 100).shift(15)


# %%
df["f_cpi_yoy"]        = cpi_daily.reindex(df.index)
df["f_cpi_trend_3m"]   = cpi_daily.diff(63).reindex(df.index)
df["f_high_inflation"] = (cpi_daily > 4).astype(int).reindex(df.index)


# %% [markdown]
# # 10 : MARKET STRESS FEATURES

# %% [markdown]
# #### 10.1  VIX Regime (ใช้ vix_close ที่ยังอยู่ใน df)

# %%
vix = df["vix_close"]

# %%
df["f_vix_regime"]    = pd.cut(vix, bins=[0, 15, 25, 35, 9999],
                                labels=[0, 1, 2, 3]).astype(float)
df["f_vix_spike"]     = (vix > vix.rolling(60).mean() * 1.5).astype(int)
df["f_vix_trend_20d"] = vix.diff(20)
df["f_vix_ma_ratio"]  = vix / vix.rolling(252).mean()

# %% [markdown]
# #### 10.2  DXY Trend

# %%
dxy_col = "f_dxy_close" if "f_dxy_close" in df.columns else None
if dxy_col:
    dxy = df[dxy_col]
else:
    # ใช้ log_gold_close เป็น proxy ไม่ได้ → ดึงจาก ret แล้ว reconstruct
    # หรือ fallback: ใช้ dxy_ret1d cumsum (approximate)
    dxy = df.get("dxy_ret1d", pd.Series(dtype=float))

if not dxy.empty:
    df["f_dxy_trend_20d"]   = dxy.diff(20)
    df["f_dxy_trend_60d"]   = dxy.diff(60)
    df["f_dxy_above_ma200"] = (dxy > dxy.rolling(200).mean()).astype(int)
else:
    print("   WARNING: DXY series ไม่พบ ข้ามการสร้าง DXY trend features")


# %% [markdown]
# #### 10.3  High Yield Spread (Credit Stress)

# %%
hy_daily = to_bday_series(
    fred.get_series("BAMLH0A0HYM2EY", observation_start=START, observation_end=END)
)

# %%
df["f_hy_spread"]         = hy_daily.reindex(df.index)
df["f_hy_spread_chg_20d"] = hy_daily.diff(20).reindex(df.index)
df["f_credit_stress"]     = (
    hy_daily > hy_daily.rolling(252).quantile(0.75)
).astype(int).reindex(df.index)


# %% [markdown]
# # 11 : GOLD MICROSTRUCTURE FEATURES

# %%
gold_ret_series = df["gold_close_ret"]

# %% [markdown]
# #### 11.1  Seasonality (cyclical encoding)

# %%
df["f_month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["f_month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
df["f_is_q1"]     = df.index.month.isin([1, 2, 3]).astype(int)
df["f_is_q4"]     = df.index.month.isin([10, 11, 12]).astype(int)


# %% [markdown]
# #### 11.2  Gold Momentum Regime

# %%
ret_1m = gold_ret_series.rolling(21).mean()
ret_3m = gold_ret_series.rolling(63).mean()

df["f_gold_momentum_regime"] = np.where(
    (ret_1m > 0) & (ret_3m > 0), 2,
    np.where((ret_1m < 0) & (ret_3m < 0), 0, 1)
)

# %% [markdown]
# #### 11.3  Gold Volatility Regime

# %%
gold_vol  = df["gold_vol_30d"]
vol_med   = gold_vol.rolling(252).median()

df["f_gold_vol_regime"] = np.where(
    gold_vol > vol_med * 1.5, 2,
    np.where(gold_vol < vol_med * 0.7, 0, 1)
)

# %% [markdown]
# # 12 : LEAKAGE CHECK

# %%
MACRO_STRESS_FEATURES = [
    "f_fed_rate", "f_fed_rate_chg_3m", "f_fed_hike_cycle",
    "f_yield_curve", "f_yield_curve_slope", "f_inverted_curve",
    "f_real_rate_10y", "f_real_rate_chg_1m", "f_real_rate_negative",
    "f_cpi_yoy", "f_cpi_trend_3m", "f_high_inflation",
    "f_vix_regime", "f_vix_spike", "f_vix_trend_20d", "f_vix_ma_ratio",
    "f_dxy_trend_20d", "f_dxy_trend_60d", "f_dxy_above_ma200",
    "f_hy_spread", "f_hy_spread_chg_20d", "f_credit_stress",
    "f_month_sin", "f_month_cos", "f_is_q1", "f_is_q4",
    "f_gold_momentum_regime", "f_gold_vol_regime",
]

# %%
checks_pass = True
for col in MACRO_STRESS_FEATURES:
    if col not in df.columns:
        print(f"   MISSING : {col}")
        checks_pass = False
        continue
    nan_pct = df[col].isnull().sum() / len(df) * 100
    status  = "OK  " if nan_pct < 10 else "WARN"
    print(f"   {status}  {col:<35} NaN: {df[col].isnull().sum():4d} ({nan_pct:.1f}%)")

print("\n   ทุก feature ผ่านการตรวจสอบ" if checks_pass else
      "\n   มี feature บางตัวที่ต้องตรวจสอบเพิ่มเติม")


# %%
# Forward/Backward fill สำหรับ NaN ช่วง warmup
df[MACRO_STRESS_FEATURES] = (
    df[MACRO_STRESS_FEATURES].ffill().bfill()
)
print(f"   NaN เหลือหลัง fill: {df[MACRO_STRESS_FEATURES].isnull().sum().sum()}")


# %% [markdown]
# # 13 : TARGET CREATION & FEATURE SHIFT

# %%
# รวม feature list ทั้งหมดที่จะ shift
ALL_FEATURES = KEEP_FEATURES + MACRO_STRESS_FEATURES

# %%
def create_target_and_shift_features(df, features, target_col="gold_close_ret"):
    df_model = df.copy()

    # Target: return วันพรุ่งนี้
    df_model["target_return"] = df_model[target_col].shift(-1)

    # Target Direction: up(1) / side(0) / down(-1) — Dynamic Threshold
    threshold = df_model["gold_vol_30d"] / np.sqrt(252) * 100 * 0.5
    df_model["target_direction"] = np.where(
        df_model["target_return"] >  threshold,  1,
        np.where(df_model["target_return"] < -threshold, -1, 0)
    )

    # Shift features ทั้งหมด 1 วัน
    shifted_features = []
    for f in features:
        if f in df_model.columns:
            new_col = f"f_{f}" if not f.startswith("f_") else f
            df_model[new_col] = df_model[f].shift(1)
            shifted_features.append(new_col)

    # Drop แถว NaN ช่วงต้น (rolling warmup)
    df_model.dropna(subset=shifted_features + ["target_return"], inplace=True)

    return df_model, shifted_features

# %%
df_model, feature_cols = create_target_and_shift_features(df, ALL_FEATURES)
print(f"   Shape หลัง shift + dropna : {df_model.shape}")
print(f"   จำนวน Features            : {len(feature_cols)}")

# %% [markdown]
# # 14 : SKLEARN PIPELINE — SCALING & SPLIT

# %%
split_date = "2024-01-01"
train = df_model[df_model.index < split_date]
test  = df_model[df_model.index >= split_date]

X_train     = train[feature_cols]
y_train     = train["target_return"]
y_train_cls = train["target_direction"]

X_test      = test[feature_cols]
y_test      = test["target_return"]
y_test_cls  = test["target_direction"]

print(f"   Train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} rows)")
print(f"   Test : {test.index.min().date()} → {test.index.max().date()} ({len(test)} rows)")


# %%
preprocessing = Pipeline([
    ("var_filter", VarianceThreshold(threshold=0.0)),
    ("scaler",     RobustScaler()),
])

# %%
X_train_scaled = preprocessing.fit_transform(X_train)
X_test_scaled  = preprocessing.transform(X_test)


# %%
print(f"\n   Final Feature Shape (Train): {X_train_scaled.shape}")
print(f"   Final Feature Shape (Test) : {X_test_scaled.shape}")


# %%
tscv = TimeSeriesSplit(n_splits=5)
print(f"   TimeSeriesSplit : {tscv.n_splits} folds")

# %% [markdown]
# # 15 : SAVE — SINGLE FINAL CSV

# %%
output_cols = feature_cols + ["target_return", "target_direction"]
df_model[output_cols].to_csv(OUT_PATH)
print(f"   Saved → {OUT_PATH}")

with open(LIST_PATH, "w") as f:
    for col in feature_cols:
        f.write(col + "\n")
print(f"   Saved → {LIST_PATH}")


# %% [markdown]
# # 16 : SUMMARY

# %%
groups = {
    "Technical / Price (Step 2–8)": [
        c for c in feature_cols
        if any(k in c for k in ["ret", "ma", "roc", "range", "body", "lag",
                                 "log_", "ratio", "spread", "corr",
                                 "day_of_week", "month", "quarter",
                                 "is_month_end", "days_gap", "rollover",
                                 "vix_close", "yield_close"])
        and not c.startswith("f_")
    ],
    "Volatility & Regime (Step 3)": [
        c for c in feature_cols
        if any(k in c for k in ["vol_", "regime", "zscore", "vix_mom"])
        and not c.startswith("f_")
    ],
    "Macro Regime — FRED (Step 9)": [
        c for c in feature_cols
        if any(k in c for k in ["fed", "yield_curve", "real_rate", "cpi", "inflation", "inverted"])
    ],
    "Market Stress (Step 10)": [
        c for c in feature_cols
        if any(k in c for k in ["vix_regime", "vix_spike", "vix_trend", "vix_ma",
                                 "dxy_trend", "dxy_above", "hy_", "credit"])
    ],
    "Gold Microstructure (Step 11)": [
        c for c in feature_cols
        if any(k in c for k in ["month_sin", "month_cos", "is_q", "momentum_regime", "vol_regime"])
    ],
}

for grp_name, cols in groups.items():
    print(f"\n  {grp_name}: {len(cols)} features")
    for col in cols[:5]:   # แสดงแค่ 5 ตัวแรกต่อกลุ่ม
        if col in df_model.columns:
            v = df_model[col].describe()
            print(f"    {col:<40} mean={v['mean']:>8.3f}  std={v['std']:>7.3f}")
    if len(cols) > 5:
        print(f"    ... และอีก {len(cols) - 5} features")

print(f"""
  ─────────────────────────────────────────────────────
  Total features  : {len(feature_cols)}
  Train rows      : {len(train)}
  Test rows       : {len(test)}
  Target (reg)    : target_return  (next-day %)
  Target (cls)    : target_direction  (-1 / 0 / 1)
  Output CSV      : {OUT_PATH}
  ─────────────────────────────────────────────────────
  ขั้นตอนต่อไป:
  → รัน Step 5 (train_test_split) บน featured_data_final.csv
  → Retrain RF และเปรียบเทียบ DA / IC กับ baseline
""")

# %%



