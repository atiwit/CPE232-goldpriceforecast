# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

# %% [markdown]
# # 1. Load & Sanity Check

# %%
df = pd.read_csv(
    "../data/processed/cleaned/cleanedoutput.csv",
    parse_dates=["Date"],
    index_col="Date"
)
df.sort_index(inplace=True)
 
print(f"Shape after load : {df.shape}")
print(f"Date range       : {df.index.min().date()} to {df.index.max().date()}")
print(f"Missing values   : {df.isnull().sum().sum()}")
 

# %% [markdown]
# # 2. Feature Creation
# 

# %% [markdown]
# 
# #### 2.1 TECHNICAL / PRICE FEATURES
# 
# - EDA insight: ราคาดิบ Non-stationary → ต้องใช้ % Return
# - EDA insight: ราคาทองมี Mean-Reversion กลับเส้น 30d MA เสมอ
#   ->สร้าง Distance-from-MA และ MA Crossover

# %%
price_cols = ["gold_close", "dxy_close", "vix_close",
              "yield_close", "sp500_close", "oil_close"]

# %%
# --- % Daily Return (เที่ยบกับวันก่อน) ---
for col in price_cols:
    ret_col = f"{col}_ret1d"
    if ret_col not in df.columns:          # ถ้ายังไม่มีจาก EDA
        df[ret_col] = df[col].pct_change() * 100

# %%
# --- Moving Averages & Distance จาก MA ---
for window in [5, 10, 20, 30, 60]:
    df[f"gold_ma{window}"]       = df["gold_close"].rolling(window).mean()
    df[f"gold_dist_ma{window}"]  = (                          # % ห่างจาก MA
        (df["gold_close"] - df[f"gold_ma{window}"]) /
         df[f"gold_ma{window}"] * 100
    )

# %%
# --- MA Crossover Signal (5d vs 20d) ---
# EDA: ราคาทองมักดึงกลับ MA → Crossover บอกสัญญาณ momentum
df["gold_ma_cross_5_20"] = (
    df["gold_ma5"] - df["gold_ma20"]
)

# %%
# --- Momentum (Rate-of-Change) ---
for lag in [1, 3, 5, 10]:
    df[f"gold_roc_{lag}d"] = df["gold_close"].pct_change(lag) * 100
 

# %%
# --- Price Range (High-Low) / Candle Body ---
# EDA: มี gold_high, gold_low, gold_open
df["gold_range_pct"] = (
    (df["gold_high"] - df["gold_low"]) / df["gold_low"] * 100
)
df["gold_body_pct"] = (
    (df["gold_close"] - df["gold_open"]) / df["gold_open"] * 100
)


# %%
# --- Lagged Returns (เพื่อป้องกัน Leakage) ---
# ต้องใช้ shift(1) ทุกครั้งก่อนส่งเข้าโมเดล
for lag in [1, 2, 3, 5]:
    df[f"gold_ret_lag{lag}"] = df["gold_close_ret"].shift(lag)
    df[f"sp500_ret_lag{lag}"] = df["sp500_close_ret"].shift(lag)
 

# %% [markdown]
# #### 2.2 VOLATILITY & REGIME FEATURES
# 
# - EDA insight: Volatility Clustering -> ใช้เป็น Feature บอก Regime
# - EDA insight: Regime Shift ปี 2024-2026 -> ต้องมี Regime Indicator
# 

# %%
# --- Rolling Volatility หลาย Window ---
for w in [5, 10, 30, 60]:
    col = f"gold_vol_{w}d"
    df[col] = df["gold_close_ret"].rolling(w).std() * np.sqrt(252) * 100

# %%
# --- sp500 Rolling Volatility (จากการ EDA) ---
for w in [7, 30]:
    df[f"sp500_vol_{w}d"] = (
        df["sp500_close_ret"].rolling(w).std() * np.sqrt(252) * 100
    )

# %%
# --- Volatility Spread: VIX − gold_vol_30d ---
# EDA insight: Spread ติดลบมากใน 2025-2026 = ทองมี Factor เฉพาะตัว
df["vol_spread_vix_gold"] = df["vix_close"] - df["gold_vol_30d"]

# %%
# --- VIX Momentum (3d & 5d) ---
# EDA insight: ความตื่นตระหนกที่ "กำลังเพิ่ม" สำคัญกว่าระดับ VIX ณ วันนั้น
df["vix_mom_3d"] = df["vix_close"].pct_change(3) * 100
df["vix_mom_5d"] = df["vix_close"].pct_change(5) * 100

# %%
# --- High-Vol Regime Indicator (Binary) ---
# EDA insight: gold_vol_30d > 25% = High Regime
df["regime_high_vol"] = (df["gold_vol_30d"] > 25).astype(int)

# %%
# --- Volatility Z-Score (ผิดปกติมากแค่ไหนเทียบ 252 วัน) ---
df["gold_vol_zscore"] = (
    (df["gold_vol_30d"] - df["gold_vol_30d"].rolling(252).mean()) /
     df["gold_vol_30d"].rolling(252).std()
)
 

# %% [markdown]
# #### 2.3 CROSS-ASSET RATIO FEATURES
# - EDA insight: Correlation ระหว่าง Gold-SP500 เปลี่ยนไปตาม Regime -> สร้าง Relative Strength Ratios

# %%
# Gold / SP500 Ratio (จากการ EDA)
df["gold_sp500_ratio"] = df["gold_close"] / df["sp500_close"]

# %%
# Oil / Gold Ratio  (proxy inflation)
df["oil_gold_ratio"]   = df["oil_close"] / df["gold_close"]

# %%
# DXY % Change (EDA: DXY ดิบใช้ได้น้อย เพราะ scale เล็ก)
df["dxy_ret1d"] = df["dxy_close"].pct_change() * 100

# %%
# Yield * DXY Interaction
# EDA: Yield & DXY มี Multicollinearity → เอา product ไว้ให้ Tree แยก Node
df["yield_x_dxy"] = df["yield_close"] * df["dxy_close"]

# %%
# Gold vs Oil (Inflation hedge signal)
df["gold_oil_spread_ret"] = df["gold_close_ret"] - df["oil_close_ret"]

# %% [markdown]
# #### 2.4 ROLLING CORRELATION FEATURES
# - EDA insight: Rolling 90d Correlation เปลี่ยนไป-มา (Non-stationary) -> ส่งค่านี้เข้าโมเดล Tree เพื่อบอก สภาวะตลาดปัจจุบัน
# - !!ต้องใช้ % Return ไม่ใช่ราคาดิบ 

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
# #### 2.5 CALENDAR / TIME FEATURES
# - EDA insight: วันที่ 26-29 ของเดือน = Contract Rollover → ควร Force Hold
# - EDA insight: วันหยุด = ช่อง gap → days_since_last_trade

# %%
df["day_of_week"]    = df.index.dayofweek          # 0=Mon … 4=Fri
df["month"]          = df.index.month
df["quarter"]        = df.index.quarter
df["is_month_end"]   = df.index.is_month_end.astype(int)

# %%
# days_since_last_trade (แก้ปัญหา .shift(1) ข้ามวันหยุด)
df["days_gap"] = df.index.to_series().diff().dt.days.fillna(1).astype(int)

# %%
# Rollover Warning Flag: day 26-29 ของเดือน
df["rollover_flag"] = df.index.day.isin([26, 27, 28, 29]).astype(int)

# %% [markdown]
# # 3. FEATURE TRANSFORMATION

# %% [markdown]
# #### 3.1 Log Transform
# - ใช้กับ: gold_close, sp500_close (right-skewed มาก skew=+2.01, +0.66)
# - ผลที่ได้: กระจายตัวสมมาตรขึ้น ช่วย Linear Regression

# %%
for col in ["gold_close", "sp500_close", "oil_close"]:
    df[f"log_{col}"] = np.log(df[col])

# %% [markdown]
# #### 3.2 Log Transform สำหรับ Volatility (right-skewed)

# %%
df["log_gold_vol_30d"] = np.log(df["gold_vol_30d"].clip(lower=0.01))
df["log_vix"]          = np.log(df["vix_close"])

# %% [markdown]
# #### 3.3 Columns ที่ไม่ต้อง Transform
# - gold_close_ret, vix_close_ret, ฯลฯ -> อยู่ใน % แล้ว ปกติพอ
# - yield_close -> ปกติ (skew=+0.15)
# - dxy_close   -> ปกติ (skew=+0.47)
#  
# #### 3.4 Scaling 
# - (ทำ ทีหลังสุดใน Pipeline เพื่อป้องกัน Leakage) -> จะทำใน sklearn Pipeline ใน Section 6
#  

# %% [markdown]
# # 4. FEATURE SELECTION & DROP

# %%
# ตัดออก: gold_vol (raw futures volume)
# เหตุผล: EDA พบ corr(vol, |return|) = -0.02 ≈ 0
# spike เกิดจาก Contract Rollover ไม่ใช่ Demand/Supply จริง
# โมเดลจะเรียน noise แทน signal
df.drop(columns=["gold_vol"], inplace=True, errors="ignore")

# %%
# ตัดออก: ราคาดิบ (Raw Prices) ถ้าจะทำ Regression / Classification
# gold_close, sp500_close, dxy_close, oil_close, vix_close, yield_close
# เหตุผล: Non-stationary → โมเดลเรียน trend ไม่ใช่ pattern
# เก็บไว้แค่: log_gold_close (สำหรับ level-based model) และ _ret columns (สำหรับ return-based model)

raw_price_cols = [
    "gold_close", "gold_high", "gold_low", "gold_open",
    "sp500_close", "dxy_close", "oil_close"
    # ไม่ตัด vix_close, yield_close เพราะใส่ใน KEEP_FEATURES แล้ว
]
df.drop(columns=raw_price_cols, inplace=True, errors="ignore")

# %%
# ตัดออก: gold_return และ sp500_return (raw, ยังไม่ shift)
# เพราะ create_target_and_shift_features() จะ shift ให้เองใน step ถัดไป
df.drop(columns=["gold_return", "sp500_return", "abs_return"],
        inplace=True, errors="ignore")

# %%
KEEP_FEATURES = [
    # --- Returns ---
    "gold_close_ret", "dxy_close_ret", "vix_close_ret",
    "yield_close_ret", "sp500_close_ret", "oil_close_ret",
    # --- Lagged Returns ---
    "gold_ret_lag1", "gold_ret_lag2", "gold_ret_lag3", "gold_ret_lag5",
    "sp500_ret_lag1", "sp500_ret_lag3",
    # --- Technical ---
    "gold_dist_ma5", "gold_dist_ma10", "gold_dist_ma20", "gold_dist_ma30",
    "gold_ma_cross_5_20",
    "gold_roc_1d", "gold_roc_3d", "gold_roc_5d", "gold_roc_10d",
    "gold_range_pct", "gold_body_pct",
    # --- Volatility & Regime ---
    "gold_vol_5d", "gold_vol_10d", "gold_vol_30d", "gold_vol_60d",
    "sp500_vol_7d", "sp500_vol_30d",
    "vol_spread_vix_gold",
    "vix_mom_3d", "vix_mom_5d",
    "regime_high_vol",
    "gold_vol_zscore",
    "log_gold_vol_30d", "log_vix",
    # --- Cross-Asset ---
    "gold_sp500_ratio", "oil_gold_ratio",
    "yield_x_dxy",
    "gold_oil_spread_ret",
    # --- Rolling Correlation ---
    "corr_90d_dxy", "corr_90d_sp500", "corr_90d_oil",
    "corr_90d_yield", "corr_90d_vix",
    # --- Calendar ---
    "day_of_week", "month", "quarter",
    "is_month_end", "days_gap", "rollover_flag",
    # --- Raw Level (เฉพาะถ้าใช้ log-price model) ---
    "log_gold_close", "log_sp500_close",
    # --- Levels ที่ยังมีประโยชน์ (stationary-enough) ---
    "vix_close", "yield_close",   # VIX stationary (ADF p=0.00)
]
 

# %%
# Sanity check หลัง drop
print("Columns remaining:", df.shape[1])

# ตรวจว่า KEEP_FEATURES ทุกตัวยังอยู่ใน df
missing_features = [f for f in KEEP_FEATURES if f not in df.columns]
if missing_features:
    print(f"Features หายไป: {missing_features}")
else:
    print("KEEP_FEATURES อยู่ครบทุกตัว")

# %% [markdown]
# # 5. DATA LEAKAGE CHECK

# %% [markdown]
# Features ที่ต้องระวัง (อาจ Leakage ถ้าไม่ระวัง):
#  
# 1. gold_close_ret  →  ถ้าใช้เป็น Target (next-day return)
#        ต้อง shift(1) ก่อนเป็น Feature ไม่ควรใช้วันเดียวกัน
#  
# 2. gold_high, gold_low, gold_open  →  Available ONLY after market close
#        ถ้าทำนายวันนี้ก่อนตลาดเปิด ห้ามใช้ค่าวันนี้ → ต้อง shift(1)
#  
# 3. gold_range_pct, gold_body_pct  →  คำนวณจาก high/low/open/close วันนี้
#        → shift(1) เสมอถ้าทำนาย next-day
#  
# 4. Rolling Correlations (corr_90d_*)  →  ใช้ได้ เพราะหา corr จาก 90 วันก่อน
#        แต่ต้อง shift(1) ด้วยเช่นกัน
#  
# 5. gold_vol_30d  →  ใช้ข้อมูลย้อนหลัง 30 วัน → ปลอดภัย แต่ต้อง shift(1)
#  
# RULE: Feature ทุกตัว ต้อง shift(1) ก่อนเข้าโมเดล
#       Target = gold_close_ret วันถัดไป (ไม่ต้อง shift)

# %%
def create_target_and_shift_features(df, features, target_col="gold_close_ret"):

    df_model = df.copy()
    
    # Target: return วันพรุ่งนี้
    df_model["target_return"]   = df_model[target_col].shift(-1)
    
    # Target Classification: up(1) / side(0) / down(-1)
    # ใช้ Dynamic Threshold จาก gold_vol_30d (EDA แนะนำ)
    threshold = df_model["gold_vol_30d"] / np.sqrt(252) * 100 * 0.5
    df_model["target_direction"] = np.where(
        df_model["target_return"] >  threshold,  1,
        np.where(df_model["target_return"] < -threshold, -1, 0)
    )
    
    # Shift features ทั้งหมด 1 วัน (ใช้ข้อมูลวานนี้ทำนายวันนี้)
    shifted_features = []
    for f in features:
        if f in df_model.columns:
            new_col = f"f_{f}"
            df_model[new_col] = df_model[f].shift(1)
            shifted_features.append(new_col)
    
    # Drop แถวที่มี NaN (ช่วงแรกที่ rolling ยังไม่มีข้อมูล)
    df_model.dropna(subset=shifted_features + ["target_return"], inplace=True)
    
    return df_model, shifted_features
 
 
df_model, feature_cols = create_target_and_shift_features(df, KEEP_FEATURES)
print(f"\nShape หลัง shift + dropna: {df_model.shape}")
print(f"จำนวน Features: {len(feature_cols)}")

# %% [markdown]
# # 6. FINAL SKLEARN PIPELINE

# %%
# Train/Test Split (ต้องใช้ Time-aware split ไม่ใช่ random)
# ใช้ข้อมูลก่อนปี 2024 เป็น Train, 2024+ เป็น Test
# (เพราะ EDA พบ Regime Shift ปี 2024-2026)
split_date = "2024-01-01"
train = df_model[df_model.index < split_date]
test  = df_model[df_model.index >= split_date]
 
X_train = train[feature_cols]
y_train = train["target_return"]       # Regression
y_train_cls = train["target_direction"] # Classification
 
X_test  = test[feature_cols]
y_test  = test["target_return"]
y_test_cls = test["target_direction"]
 
print(f"\nTrain: {train.index.min().date()} → {train.index.max().date()} ({len(train)} rows)")
print(f"Test : {test.index.min().date()} → {test.index.max().date()} ({len(test)} rows)")

# %%
# Preprocessing Pipeline
preprocessing = Pipeline([
    ("var_filter", VarianceThreshold(threshold=0.0)),   # ตัด constant columns
    ("scaler",     RobustScaler()),                      # robust ต่อ outlier
])

# %%
# Fit บน Train เท่านั้น → Transform ทั้ง Train และ Test
X_train_scaled = preprocessing.fit_transform(X_train)
X_test_scaled  = preprocessing.transform(X_test)
 
print(f"\nFinal Feature Shape (Train): {X_train_scaled.shape}")
print(f"Final Feature Shape (Test) : {X_test_scaled.shape}")
 

# %%
# TimeSeriesSplit สำหรับ Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
print(f"\nTimeSeriesSplit กำหนด {tscv.n_splits} folds")

# %% [markdown]
# # 7. SAVE

# %%
# บันทึก feature dataset (ยังไม่ scaled เพื่อให้ inspect ได้)
output_cols = feature_cols + ["target_return", "target_direction"]
df_model[output_cols].to_csv(
    "../data/processed/featured/featured_data.csv"

)
print("\nSaved featured_data.csv")
 

# %%
# บันทึก feature list
with open("../data/processed/featured/feature_list.txt", "w") as f:
    for col in feature_cols:
        f.write(col + "\n")
print("Saved feature_list.txt")

# %%
print("\n" + "="*60)
print("  SUMMARY: Feature Engineering Complete")
print("="*60)
print(f"  Total features created : {len(feature_cols)}")
print(f"  Train rows             : {len(train)}")
print(f"  Test rows              : {len(test)}")
print(f"  Target (regression)    : target_return (next-day %)")
print(f"  Target (classification): target_direction (-1/0/1)")
print("="*60)
 

# %%



