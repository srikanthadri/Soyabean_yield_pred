# -*- coding: utf-8 -*-
"""
Soybean Yield Forecasting Dashboard (Streamlit)
✅ Builds master feature table from multiple CSVs (weather + LAI/GPP long-format + yield)
✅ Fixes duplicates (many-to-many merges) by aggregating to ONE row per State-District-Year
✅ Trains RF model (year range controls), removes outliers, feature engineering
✅ Predicts selected year
✅ Shows Folium district map + click-to-view district details
✅ Shows scatter, feature importance, and simple trends

Author: (you)
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from functools import reduce

import folium
from streamlit_folium import st_folium

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================================
# 0. PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Soybean Yield Forecast Dashboard", layout="wide")
st.title("🌾 Soybean Yield Forecast Dashboard")

st.markdown(
    """
This app:
- Reads **multiple weather CSVs** (PRCP/TMAX/TMIN/WIND/RH/SRAD)
- Reads **LAI & GPP long-format CSVs** (month column)
- Reads **Yield CSV**
- Builds a **single clean master table** (one row per State–District–Year)
- Trains RandomForest, shows metrics + scatter + feature importance
- Predicts yield for selected year
- Shows **district map** with click-to-view details
"""
)

st.markdown(
    """
<style>
p, li { font-size: 1.02rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 1. INPUTS (SIDEBAR)
# ==========================================================
st.sidebar.header("📁 Input Paths")

DEFAULT_BASE = r"F:\COMSNAP_Sureshsir\soya\modeltest\yiled"
BASE_DIR = st.sidebar.text_input("Base folder", DEFAULT_BASE)

file_prcp = str(Path(BASE_DIR) / "WS_Prcp_2021-25_monsum.csv")
file_tmax = str(Path(BASE_DIR) / "WS_tmax_2021-25_monmean.csv")
file_tmin = str(Path(BASE_DIR) / "WS_tmin_2021-25_monmean.csv")
file_wind = str(Path(BASE_DIR) / "WS_wind_2021-25_monmean.csv")
file_rh   = str(Path(BASE_DIR) / "WS_rhavg_2021-25_monmean.csv")
file_srad = str(Path(BASE_DIR) / "WS_srad_2021-25_monsum.csv")

file_yield = str(Path(BASE_DIR) / "soya_stats2021_25_all_states_yield.csv")
file_lai   = str(Path(BASE_DIR) / "Soybean_LAImax_monthly_district_2021_2025.csv")
file_gpp   = str(Path(BASE_DIR) / "Soybean_GPPsum_monthly_district_2021_2025.csv")

DEFAULT_SHP = str(Path(BASE_DIR) / "3states.shp")
shp_path = st.sidebar.text_input("District shapefile (.shp)", DEFAULT_SHP)

st.sidebar.header("🗓️ Year Controls")
train_start = st.sidebar.number_input("Train start year", value=2021, min_value=1900, max_value=2100, step=1)
train_end   = st.sidebar.number_input("Train end year",   value=2024, min_value=1900, max_value=2100, step=1)
predict_year = st.sidebar.number_input("Predict year", value=2025, min_value=1900, max_value=2100, step=1)

months = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]  # as you asked: 6 to 11

# ==========================================================
# 2. HELPERS
# ==========================================================
def norm_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\u00a0", " ", regex=False).str.strip().str.lower()

def safe_read_csv(path, encodings=("utf-8", "utf-8-sig", "latin1", "cp1252")):
    """Try multiple encodings to avoid UnicodeDecodeError."""
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def ensure_cols(df, cols, name="file"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Available: {df.columns.tolist()}")

def load_weather(file_path, var_prefix, months):
    """
    Expects columns like: State, District, Year, Jun, Jul, ...
    Drops Unq/data if present.
    Aggregates duplicates to ONE row per State-District-Year (mean).
    """
    df = safe_read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # drop helpers
    for col in ["Unq", "data"]:
        if col in df.columns:
            df = df.drop(columns=col)

    ensure_cols(df, ["State", "District", "Year"], name=file_path)

    keep_months = [m for m in months if m in df.columns]
    df = df[["State", "District", "Year"] + keep_months].copy()

    # normalize join keys
    df["State"] = norm_text(df["State"])
    df["District"] = norm_text(df["District"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # rename month columns
    rename_map = {m: f"{var_prefix}_{m}" for m in keep_months}
    df = df.rename(columns=rename_map)

    # IMPORTANT: aggregate duplicates -> one row
    agg_cols = [c for c in df.columns if c not in ["State", "District", "Year"]]
    df = df.groupby(["State", "District", "Year"], as_index=False)[agg_cols].mean()

    return df

def load_yield(file_path):
    """
    Expects: State, District, Year, Yield (+ optional Acreage)
    Aggregates duplicates to ONE row per State-District-Year:
      - Yield: mean (or max if you prefer)
      - Acreage: sum (usually acreage is additive)
    """
    df = safe_read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    ensure_cols(df, ["State", "District", "Year"], name=file_path)

    # detect yield column
    ycol = "Yield" if "Yield" in df.columns else None
    if ycol is None:
        raise ValueError(f"{file_path} must have 'Yield' column. Found: {df.columns.tolist()}")

    # optional acreage
    acre_col = None
    for c in ["Acreage", "Area", "acreage"]:
        if c in df.columns:
            acre_col = c
            break

    keep = ["State", "District", "Year", ycol] + ([acre_col] if acre_col else [])
    df = df[keep].copy()

    df["State"] = norm_text(df["State"])
    df["District"] = norm_text(df["District"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")

    if acre_col:
        df[acre_col] = pd.to_numeric(df[acre_col], errors="coerce")

        df = df.groupby(["State", "District", "Year"], as_index=False).agg(
            Yield=("Yield", "mean"),
            Acreage=(acre_col, "sum")
        )
    else:
        df = df.groupby(["State", "District", "Year"], as_index=False).agg(
            Yield=("Yield", "mean")
        )

    return df

def load_lai_or_gpp_long(file_path, prefix, months, month_nums=(6,7,8,9,10,11)):
    """
    LAI/GPP long-format: has month column (1-12), plus State/District/Year and a value column.
    Returns wide: prefix_Jun ... prefix_Nov.
    Aggregates duplicates per month if present.
    """
    df = safe_read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # detect cols
    state_col = "State" if "State" in df.columns else ("state_name" if "state_name" in df.columns else None)
    dist_col  = "District" if "District" in df.columns else ("district_name" if "district_name" in df.columns else None)
    year_col  = "Year" if "Year" in df.columns else ("year" if "year" in df.columns else None)

    if state_col is None or dist_col is None or year_col is None:
        raise ValueError(f"{file_path}: cannot find State/District/Year columns. Got: {df.columns.tolist()}")

    if "month" not in df.columns:
        raise ValueError(f"{file_path}: expected 'month' column for long-format LAI/GPP.")

    # find value col
    value_col = None
    for c in df.columns:
        lc = c.lower()
        if prefix.lower().startswith("lai") and "lai" in lc:
            value_col = c; break
        if prefix.lower().startswith("gpp") and "gpp" in lc:
            value_col = c; break
    if value_col is None:
        raise ValueError(f"{file_path}: cannot find value column for {prefix}. Columns: {df.columns.tolist()}")

    df = df[[state_col, dist_col, year_col, "month", value_col]].copy()
    df = df.rename(columns={state_col:"State", dist_col:"District", year_col:"Year"})

    df["State"] = norm_text(df["State"])
    df["District"] = norm_text(df["District"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # filter months 6..11
    df = df[df["month"].isin(list(month_nums))].copy()

    month_map = {6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov"}
    df["MonthName"] = df["month"].map(month_map)

    # aggregate duplicates (State, District, Year, MonthName)
    df = df.groupby(["State","District","Year","MonthName"], as_index=False)[value_col].mean()

    wide = (
        df.pivot_table(index=["State","District","Year"], columns="MonthName", values=value_col, aggfunc="mean")
          .reset_index()
    )

    # rename month columns to prefix_Mon
    for m in months:
        if m in wide.columns:
            wide = wide.rename(columns={m: f"{prefix}_{m}"})

    # final aggregate safeguard
    val_cols = [c for c in wide.columns if c not in ["State","District","Year"]]
    wide = wide.groupby(["State","District","Year"], as_index=False)[val_cols].mean()

    return wide

def add_engineered_features(df, months):
    """Feature engineering similar to your notebook."""
    # seasonal sums/means
    prcp_cols = [f"PRCP_{m}" for m in months if f"PRCP_{m}" in df.columns]
    tmax_cols = [f"TMAX_{m}" for m in months if f"TMAX_{m}" in df.columns]
    tmin_cols = [f"TMIN_{m}" for m in months if f"TMIN_{m}" in df.columns]
    wind_cols = [f"WIND_{m}" for m in months if f"WIND_{m}" in df.columns]
    rh_cols   = [f"RH_{m}"   for m in months if f"RH_{m}"   in df.columns]
    srad_cols = [f"SRAD_{m}" for m in months if f"SRAD_{m}" in df.columns]
    lai_cols  = [f"LAImax_{m}" for m in months if f"LAImax_{m}" in df.columns]
    gpp_cols  = [f"GPPsum_{m}" for m in months if f"GPPsum_{m}" in df.columns]

    if prcp_cols:
        df["PRCP_season"] = df[prcp_cols].sum(axis=1, skipna=True)
        ja = [c for c in prcp_cols if c.endswith("_Jul") or c.endswith("_Aug")]
        if ja:
            df["PRCP_JulAug_frac"] = np.where(df["PRCP_season"] > 0, df[ja].sum(axis=1, skipna=True) / df["PRCP_season"], np.nan)

    if tmax_cols: df["TMAX_season_mean"] = df[tmax_cols].mean(axis=1, skipna=True)
    if tmin_cols: df["TMIN_season_mean"] = df[tmin_cols].mean(axis=1, skipna=True)
    if wind_cols: df["WIND_season_mean"] = df[wind_cols].mean(axis=1, skipna=True)

    if rh_cols:
        df["RH_mean"] = df[rh_cols].mean(axis=1, skipna=True)
        df["RH_std"]  = df[rh_cols].std(axis=1, skipna=True)

    if srad_cols:
        df["SRAD_mean"] = df[srad_cols].mean(axis=1, skipna=True)
        df["SRAD_sum"]  = df[srad_cols].sum(axis=1, skipna=True)

    if lai_cols:
        df["LAImax_peak"] = df[lai_cols].max(axis=1)
        df["LAImax_sum"]  = df[lai_cols].sum(axis=1, skipna=True)

    if gpp_cols:
        df["GPPsum_season"] = df[gpp_cols].sum(axis=1, skipna=True)

    # extra: temperature range
    if tmax_cols and tmin_cols:
        df["T_range_mean"] = df[tmax_cols].mean(axis=1, skipna=True) - df[tmin_cols].mean(axis=1, skipna=True)

    return df

def build_feature_cols(df, months):
    """Create feature list robustly (includes RH/SRAD)."""
    feature_cols = []
    for v in ["PRCP","TMAX","TMIN","WIND","RH","SRAD","LAImax","GPPsum"]:
        for m in months:
            c = f"{v}_{m}"
            if c in df.columns:
                feature_cols.append(c)

    engineered = [
        "PRCP_season","PRCP_JulAug_frac","TMAX_season_mean","TMIN_season_mean","WIND_season_mean",
        "RH_mean","RH_std","SRAD_mean","SRAD_sum","LAImax_peak","LAImax_sum","GPPsum_season","T_range_mean"
    ]
    for c in engineered:
        if c in df.columns:
            feature_cols.append(c)

    return feature_cols

def iqr_outlier_filter(df, ycol="Yield"):
    y = df[ycol].dropna().values
    if len(y) < 10:
        return df, None
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    out = df[(df[ycol] >= lo) & (df[ycol] <= hi)].copy()
    return out, (lo, hi)

# ==========================================================
# 3. BUILD MASTER TABLE
# ==========================================================
st.header("1) Build Master Feature Table")

run_build = st.button("🔄 Build / Refresh Master Table")

@st.cache_data(show_spinner=True)
def build_master():
    # weather
    df_prcp = load_weather(file_prcp, "PRCP", months)
    df_tmax = load_weather(file_tmax, "TMAX", months)
    df_tmin = load_weather(file_tmin, "TMIN", months)
    df_wind = load_weather(file_wind, "WIND", months)
    df_rh   = load_weather(file_rh,   "RH",   months)
    df_srad = load_weather(file_srad, "SRAD", months)

    # yield
    df_y = load_yield(file_yield)

    # LAI/GPP long
    df_lai = load_lai_or_gpp_long(file_lai, "LAImax", months, month_nums=(6,7,8,9,10,11))
    df_gpp = load_lai_or_gpp_long(file_gpp, "GPPsum", months, month_nums=(6,7,8,9,10,11))

    # merge all (left = yield so we keep yield rows)
    dfs = [df_y, df_prcp, df_tmax, df_tmin, df_wind, df_rh, df_srad, df_lai, df_gpp]
    master = reduce(lambda L, R: pd.merge(L, R, on=["State","District","Year"], how="left"), dfs)

    # final de-dup safeguard
    numeric_cols = [c for c in master.columns if c not in ["State","District","Year"]]
    master = master.groupby(["State","District","Year"], as_index=False)[numeric_cols].mean()

    # engineered
    master = add_engineered_features(master, months)

    return master

if run_build:
    st.cache_data.clear()

try:
    master = build_master()
except Exception as e:
    st.error(f"❌ Failed building master: {e}")
    st.stop()

st.success(f"✅ Master table ready: {master.shape[0]} rows × {master.shape[1]} columns")
st.dataframe(master.head(30), use_container_width=True)

out_master = str(Path(BASE_DIR) / "soy_yield_features_master.csv")
if st.button("💾 Save master CSV"):
    master.to_csv(out_master, index=False, float_format="%.6f")
    st.success(f"Saved: {out_master}")

# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
st.header("2) Train Yield Model")

feature_cols = build_feature_cols(master, months)
st.write(f"Features used: **{len(feature_cols)}**")

df_train = master[
    (master["Year"] >= train_start) &
    (master["Year"] <= train_end) &
    master["Yield"].notna()
].copy()

if df_train.empty:
    st.warning("No training data in the selected year range.")
    st.stop()

st.write("Training rows before outlier removal:", len(df_train))
df_train_f, bounds = iqr_outlier_filter(df_train, "Yield")
st.write("Training rows after outlier removal:", len(df_train_f))
if bounds:
    st.caption(f"IQR filter bounds: {bounds[0]:.2f} to {bounds[1]:.2f}")

# fill missing features
X_all = df_train_f[feature_cols].copy()
X_all = X_all.replace([np.inf, -np.inf], np.nan)
X_all = X_all.fillna(X_all.median(numeric_only=True))

y_all = df_train_f["Yield"].astype(float).values

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.25, 0.05)

X_tr, X_te, y_tr, y_te = train_test_split(X_all.values, y_all, test_size=test_size, random_state=42)

n_estimators = st.sidebar.slider("RF trees (n_estimators)", 100, 1500, 600, 50)

model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=0,
    n_jobs=-1
)

with st.spinner("Training RandomForest..."):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
r2 = float(r2_score(y_te, y_pred))

c1, c2 = st.columns(2)
c1.metric("Test RMSE", f"{rmse:.2f}")
c2.metric("Test R²", f"{r2:.3f}")

# Scatter plot
st.subheader("Observed vs Predicted (Test Split)")
fig = plt.figure(figsize=(6, 6))
plt.scatter(y_te, y_pred, alpha=0.7)
vmin = min(np.min(y_te), np.min(y_pred))
vmax = max(np.max(y_te), np.max(y_pred))
plt.plot([vmin, vmax], [vmin, vmax], "r--")
plt.xlabel("Observed Yield")
plt.ylabel("Predicted Yield")
plt.title(f"R²={r2:.3f}, RMSE={rmse:.2f}")
plt.grid(alpha=0.3)
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
imp = imp.sort_values("importance", ascending=False).head(25)

fig2 = plt.figure(figsize=(8, 6))
plt.barh(imp["feature"][::-1], imp["importance"][::-1])
plt.xlabel("Importance")
plt.title("Top 25 Features")
plt.tight_layout()
st.pyplot(fig2)

# ==========================================================
# 5. PREDICT YEAR
# ==========================================================
st.header("3) Predict for a Year")

df_pred = master[master["Year"] == predict_year].copy()
if df_pred.empty:
    st.warning(f"No rows found for predict_year={predict_year}")
    st.stop()

Xf = df_pred[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
Xf = Xf.fillna(X_all.median(numeric_only=True))

df_pred["Predicted_Yield"] = model.predict(Xf.values)

st.success(f"✅ Predictions generated for {predict_year}: {len(df_pred)} districts")
st.dataframe(df_pred[["State","District","Year","Yield","Predicted_Yield"]].head(50), use_container_width=True)

out_pred = str(Path(BASE_DIR) / f"soy_yield_predictions_{predict_year}.csv")
if st.button("💾 Save predictions CSV"):
    df_pred.to_csv(out_pred, index=False, float_format="%.6f")
    st.success(f"Saved: {out_pred}")

# ==========================================================
# 6. MAP
# ==========================================================
st.header("4) District Map (Prediction)")

# Load shapefile
try:
    gdf = gpd.read_file(shp_path)
except Exception as e:
    st.error(f"Cannot read shapefile: {e}")
    st.stop()

# Detect district/state columns in shapefile
# (edit these if needed)
shp_dist = "District" if "District" in gdf.columns else None
shp_state = "State" if "State" in gdf.columns else None

if shp_dist is None:
    st.error(f"Shapefile must have a 'District' column. Found: {gdf.columns.tolist()}")
    st.stop()

gdf["District_key"] = norm_text(gdf[shp_dist])
if shp_state and shp_state in gdf.columns:
    gdf["State_key"] = norm_text(gdf[shp_state])
else:
    gdf["State_key"] = ""

df_map = df_pred.copy()
df_map["District_key"] = norm_text(df_map["District"])
df_map["State_key"] = norm_text(df_map["State"])

# Join: prefer both keys if state exists in shapefile
if shp_state and shp_state in gdf.columns:
    gdf_join = gdf.merge(df_map, on=["State_key","District_key"], how="left")
else:
    gdf_join = gdf.merge(df_map, on=["District_key"], how="left")

# Filter state
state_list = sorted(df_pred["State"].dropna().unique().tolist())
sel_state = st.selectbox("State filter (map)", options=["All"] + state_list, index=0)

view = gdf_join.copy()
if sel_state != "All":
    view = view[norm_text(view["State"]) == sel_state.strip().lower()] if "State" in view.columns else view

if view.empty:
    st.warning("No features to display for this state selection.")
    st.stop()

# Create folium map
bounds = view.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="cartodbpositron")

# Color scale
vmin = float(np.nanmin(view["Predicted_Yield"].values)) if "Predicted_Yield" in view.columns else 0.0
vmax = float(np.nanmax(view["Predicted_Yield"].values)) if "Predicted_Yield" in view.columns else 1.0

def yield_color(val):
    # simple 5-bin palette
    if pd.isna(val):
        return "#cccccc"
    r = (val - vmin) / (vmax - vmin + 1e-9)
    if r < 0.2: return "#2c7bb6"
    if r < 0.4: return "#abd9e9"
    if r < 0.6: return "#ffffbf"
    if r < 0.8: return "#fdae61"
    return "#d7191c"

def style_fn(feature):
    val = feature["properties"].get("Predicted_Yield", None)
    return {
        "fillColor": yield_color(val),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.75
    }

tooltip_fields = []
tooltip_aliases = []
for f, a in [
    (shp_state, "State"),
    (shp_dist, "District"),
    ("Yield", "Observed Yield"),
    ("Predicted_Yield", "Predicted Yield")
]:
    if f and f in view.columns:
        tooltip_fields.append(f)
        tooltip_aliases.append(a)

gj = folium.GeoJson(
    data=view.to_json(),
    style_function=style_fn,
    highlight_function=lambda x: {"weight": 3, "color": "blue"},
    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=False)
)
gj.add_to(m)

map_data = st_folium(m, width="100%", height=560, key="yield_map")

# Click-to-detail
st.subheader("District Detail")

all_districts = sorted(df_pred["District"].dropna().unique().tolist())
if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = all_districts[0] if all_districts else None

clicked = None
if map_data and map_data.get("last_active_drawing"):
    props = map_data["last_active_drawing"].get("properties", {})
    # try multiple district keys
    clicked = props.get(shp_dist) or props.get("District") or props.get("district")

if clicked:
    st.session_state["selected_district"] = str(clicked).strip().lower()

sel = st.session_state["selected_district"]

# find row in df_pred
row = df_pred[norm_text(df_pred["District"]) == str(sel)].head(1)
if row.empty:
    st.info("Click a district on the map to see details.")
else:
    rr = row.iloc[0]
    cA, cB = st.columns(2)
    with cA:
        st.write(f"**State:** {rr['State']}")
        st.write(f"**District:** {rr['District']}")
        st.write(f"**Year:** {int(rr['Year'])}")
    with cB:
        st.write(f"**Observed Yield:** {rr.get('Yield', np.nan)}")
        st.write(f"**Predicted Yield:** {rr.get('Predicted_Yield', np.nan):.2f}")

st.caption("✅ Tip: If map still looks blank, it means State/District names between CSV and shapefile are not matching after normalization.")
