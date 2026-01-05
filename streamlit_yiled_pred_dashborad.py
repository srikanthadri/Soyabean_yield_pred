# streamlit_yield_map_dashboard.py
# ------------------------------------------------------------
# Soybean Yield Dashboard (Map + Click District + Small Graphs)
# - District shapefile map colored by Predicted 2025 Yield
# - Click district on map -> show KPIs + compact monthly plots below
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

import folium
from streamlit_folium import st_folium
import branca.colormap as cm


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Soybean Yield + Drivers (Map)", layout="wide")
st.title("🗺️ Soybean Yield Forecast & Drivers Dashboard")
st.caption("District map colored by **Predicted Yield (2025)**. Click a district to view compact weather/phenology profiles (Hist mean vs 2024 vs 2025).")


# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("📁 Inputs")

# Prefer relative paths for GitHub/Streamlit Cloud
DEFAULT_DATA = "soy_yield_features_new_rs.csv"
DEFAULT_PRED = "soy_yield_predictions_2025.csv"   # optional (can be blank if predictions already inside master)
DEFAULT_SHP  = "3states.shp"                   # your district shapefile (or GeoJSON)

data_path = st.sidebar.text_input("Master features CSV", DEFAULT_DATA)
pred_path = st.sidebar.text_input("Predictions CSV (optional)", DEFAULT_PRED)
shp_path  = st.sidebar.text_input("District shapefile path", DEFAULT_SHP)

st.sidebar.header("📅 Year Settings")
HIST_START = int(st.sidebar.number_input("Hist start year", value=2021, step=1))
HIST_END   = int(st.sidebar.number_input("Hist end year", value=2024, step=1))
LAST_YEAR  = int(st.sidebar.number_input("Last year", value=2024, step=1))
CURR_YEAR  = int(st.sidebar.number_input("Current year", value=2025, step=1))

MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]


# -----------------------------
# HELPERS
# -----------------------------
def norm(s: pd.Series) -> pd.Series:
    """Normalize strings for safe joins."""
    return (s.astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[^A-Z0-9 ]", "", regex=True))

def safe_exists(path_str: str) -> bool:
    try:
        return Path(path_str).exists()
    except Exception:
        return False

def to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def plot_monthly_small(df_dist: pd.DataFrame, var_prefix: str, title: str, ylabel: str):
    """Small monthly profile plot: Hist mean vs 2024 vs 2025."""
    cols = [f"{var_prefix}_{m}" for m in MONTHS if f"{var_prefix}_{m}" in df_dist.columns]
    if not cols:
        st.info(f"Missing monthly columns for {var_prefix}_Jun..Nov")
        return

    hist = df_dist[(df_dist["Year"] >= HIST_START) & (df_dist["Year"] <= HIST_END)]
    last = df_dist[df_dist["Year"] == LAST_YEAR]
    curr = df_dist[df_dist["Year"] == CURR_YEAR]

    hist_mean = hist[cols].mean(axis=0, skipna=True) if not hist.empty else pd.Series([np.nan]*len(cols), index=cols)
    last_val  = last[cols].iloc[0] if len(last) else pd.Series([np.nan]*len(cols), index=cols)
    curr_val  = curr[cols].iloc[0] if len(curr) else pd.Series([np.nan]*len(cols), index=cols)

    x = [c.split("_")[-1] for c in cols]

    fig, ax = plt.subplots(figsize=(4.6, 2.5))  # SMALL
    ax.plot(x, hist_mean.values, marker="o", linewidth=1.3, label=f"Hist mean ({HIST_START}-{HIST_END})")
    if last_val.notna().any():
        ax.plot(x, last_val.values, marker="o", linewidth=1.3, label=str(LAST_YEAR))
    if curr_val.notna().any():
        ax.plot(x, curr_val.values, marker="o", linewidth=1.3, label=str(CURR_YEAR))

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best", frameon=False)
    plt.tight_layout()
    st.pyplot(fig)

def plot_yearly_small(df_dist: pd.DataFrame, ycol: str, title: str, ylabel: str):
    if ycol not in df_dist.columns:
        st.info(f"Missing column: {ycol}")
        return
    d = df_dist[["Year", ycol]].dropna().sort_values("Year")
    if d.empty:
        st.info(f"No data for {ycol}")
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.4))  # SMALL
    ax.plot(d["Year"].values, d[ycol].values, marker="o", linewidth=1.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_master(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "State" not in df.columns or "District" not in df.columns or "Year" not in df.columns:
        raise ValueError("Master CSV must contain: State, District, Year")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Yield" in df.columns:
        df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

    df["State"] = df["State"].astype(str).str.strip()
    df["District"] = df["District"].astype(str).str.strip()
    df["State_key"] = norm(df["State"])
    df["District_key"] = norm(df["District"])
    return df

@st.cache_data
def load_preds(pred_csv: str) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    if "State" not in pred.columns or "District" not in pred.columns or "Year" not in pred.columns:
        raise ValueError("Prediction CSV must contain: State, District, Year")
    pred["Year"] = pd.to_numeric(pred["Year"], errors="coerce")
    pred["State"] = pred["State"].astype(str).str.strip()
    pred["District"] = pred["District"].astype(str).str.strip()
    pred["State_key"] = norm(pred["State"])
    pred["District_key"] = norm(pred["District"])
    return pred

@st.cache_data
def load_shape(shp: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp)
    # Expect columns: District (and optionally State)
    if "District" not in gdf.columns:
        # Try common alternatives
        for alt in ["DISTRICT", "district", "Dist", "DIST"]:
            if alt in gdf.columns:
                gdf = gdf.rename(columns={alt: "District"})
                break
    if "District" not in gdf.columns:
        raise ValueError("Shapefile must contain a District column (or rename it to 'District').")

    if "State" not in gdf.columns:
        for alt in ["STATE", "state", "St", "ST"]:
            if alt in gdf.columns:
                gdf = gdf.rename(columns={alt: "State"})
                break

    gdf["District"] = gdf["District"].astype(str).str.strip()
    gdf["District_key"] = norm(gdf["District"])
    if "State" in gdf.columns:
        gdf["State"] = gdf["State"].astype(str).str.strip()
        gdf["State_key"] = norm(gdf["State"])
    else:
        gdf["State_key"] = ""

    return gdf

try:
    if not safe_exists(data_path):
        st.error(f"Master CSV not found: {data_path}")
        st.stop()
    df = load_master(data_path)

    # Bring predictions into df (if present either in df OR in pred csv)
    if "Predicted_2025_Yield" not in df.columns and "Predicted_Yield" in df.columns:
        df["Predicted_2025_Yield"] = pd.to_numeric(df["Predicted_Yield"], errors="coerce")

    if safe_exists(pred_path):
        pred = load_preds(pred_path)
        # find prediction column
        pred_col = None
        for c in ["Predicted_Yield", "Predicted_2025_Yield", "y_pred", "PredictedYield"]:
            if c in pred.columns:
                pred_col = c
                break
        if pred_col is None:
            st.warning("Prediction CSV loaded, but no prediction column found. Expected one of: Predicted_Yield / Predicted_2025_Yield")
        else:
            pred_small = pred.loc[pred["Year"] == CURR_YEAR, ["State_key", "District_key", "Year", pred_col]].copy()
            pred_small = pred_small.rename(columns={pred_col: "Predicted_2025_Yield"})
            df = df.merge(pred_small, on=["State_key", "District_key", "Year"], how="left")

    if not safe_exists(shp_path):
        st.error(f"Shapefile not found: {shp_path}")
        st.stop()
    gdf = load_shape(shp_path)

except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()


# -----------------------------
# FEATURE ENGINEERING (monthly -> seasonal helpers used in comparisons)
# -----------------------------
prcp_cols = [f"PRCP_{m}" for m in MONTHS if f"PRCP_{m}" in df.columns]
tmax_cols = [f"TMAX_{m}" for m in MONTHS if f"TMAX_{m}" in df.columns]
tmin_cols = [f"TMIN_{m}" for m in MONTHS if f"TMIN_{m}" in df.columns]
wind_cols = [f"WIND_{m}" for m in MONTHS if f"WIND_{m}" in df.columns]
rh_cols   = [f"RH_{m}"   for m in MONTHS if f"RH_{m}" in df.columns]
srad_cols = [f"SRAD_{m}" for m in MONTHS if f"SRAD_{m}" in df.columns]
lai_cols  = [f"LAImax_{m}" for m in MONTHS if f"LAImax_{m}" in df.columns]
gpp_cols  = [f"GPPsum_{m}" for m in MONTHS if f"GPPsum_{m}" in df.columns]

to_num(df, prcp_cols + tmax_cols + tmin_cols + wind_cols + rh_cols + srad_cols + lai_cols + gpp_cols)
if "Predicted_2025_Yield" in df.columns:
    df["Predicted_2025_Yield"] = pd.to_numeric(df["Predicted_2025_Yield"], errors="coerce")


# -----------------------------
# BUILD MAP DATA (join predictions to shapefile)
# -----------------------------
df_2025 = df[df["Year"] == CURR_YEAR].copy()
if df_2025.empty:
    st.error(f"No rows found for year = {CURR_YEAR} in master data.")
    st.stop()

if "Predicted_2025_Yield" not in df_2025.columns or df_2025["Predicted_2025_Yield"].notna().sum() == 0:
    st.error("Predicted_2025_Yield is missing/empty. Make sure you merged predictions CSV or have prediction column in master.")
    st.stop()

map_join = df_2025[["District_key", "Predicted_2025_Yield"]].dropna().groupby("District_key", as_index=False).mean()

gdf_map = gdf.merge(map_join, on="District_key", how="left")

# Sidebar: state filter from shapefile (if present)
st.sidebar.header("🧭 Filters")
if "State" in gdf_map.columns:
    state_list = sorted(gdf_map["State"].dropna().unique().tolist())
    sel_state = st.sidebar.selectbox("Select State (map)", ["All"] + state_list, index=0)
    if sel_state != "All":
        gdf_map_view = gdf_map[gdf_map["State"] == sel_state].copy()
        df_view = df[df["State"] == sel_state].copy()
    else:
        gdf_map_view = gdf_map.copy()
        df_view = df.copy()
else:
    sel_state = "All"
    gdf_map_view = gdf_map.copy()
    df_view = df.copy()

# District dropdown (fallback)
district_list = sorted(df_view["District"].dropna().unique().tolist())
if not district_list:
    st.error("No districts found after filtering.")
    st.stop()


# -----------------------------
# MAP SECTION (colored by predicted yield)
# -----------------------------
st.subheader(f"🗺️ Predicted Yield Map ({CURR_YEAR})")

valid_vals = gdf_map_view["Predicted_2025_Yield"].dropna()
if valid_vals.empty:
    st.warning("No predicted yield values found after joining shapefile. Check district names/keys.")
    st.stop()

ymin, ymax = float(valid_vals.min()), float(valid_vals.max())
colormap = cm.linear.YlGnBu_09.scale(ymin, ymax)
colormap.caption = f"Predicted Yield {CURR_YEAR}"

bounds = gdf_map_view.total_bounds  # [minx, miny, maxx, maxy]
center_lat = float((bounds[1] + bounds[3]) / 2)
center_lon = float((bounds[0] + bounds[2]) / 2)

m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="cartodbpositron")

def style_fn(feature):
    val = feature["properties"].get("Predicted_2025_Yield")
    if val is None:
        return {"fillColor": "#CCCCCC", "color": "black", "weight": 0.4, "fillOpacity": 0.6}
    try:
        return {"fillColor": colormap(float(val)), "color": "black", "weight": 0.4, "fillOpacity": 0.78}
    except Exception:
        return {"fillColor": "#CCCCCC", "color": "black", "weight": 0.4, "fillOpacity": 0.6}

tooltip_fields = ["District", "Predicted_2025_Yield"]
tooltip_alias  = ["District", f"Predicted Yield ({CURR_YEAR})"]
if "State" in gdf_map_view.columns:
    tooltip_fields = ["State"] + tooltip_fields
    tooltip_alias  = ["State"] + tooltip_alias

folium.GeoJson(
    gdf_map_view.to_json(),
    stylef=None,
    style_function=style_fn,
    highlight_function=lambda x: {"weight": 2, "color": "blue"},
    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_alias, sticky=False),
    name="Predicted Yield",
).add_to(m)

colormap.add_to(m)

map_data = st_folium(m, width="100%", height=460, key="yield_map")


# -----------------------------
# DISTRICT SELECTION FROM CLICK (fallback to dropdown)
# -----------------------------
clicked_district = None
if map_data and map_data.get("last_active_drawing"):
    props = map_data["last_active_drawing"].get("properties", {})
    clicked_district = props.get("District")

if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = district_list[0]

if clicked_district:
    st.session_state["selected_district"] = str(clicked_district).strip()

sel_district = st.sidebar.selectbox(
    "Select District (fallback)",
    district_list,
    index=max(0, district_list.index(st.session_state["selected_district"])) if st.session_state["selected_district"] in district_list else 0
)

# Keep session synced with dropdown too
st.session_state["selected_district"] = sel_district


# -----------------------------
# FILTER DISTRICT DATA
# -----------------------------
df_dist = df_view[df_view["District"] == sel_district].copy()
df_dist = df_dist.sort_values("Year")

if df_dist.empty:
    st.warning("No rows for selected district.")
    st.stop()

# KPI values
hist = df_dist[(df_dist["Year"] >= HIST_START) & (df_dist["Year"] <= HIST_END)]
mean_yield = hist["Yield"].mean(skipna=True) if "Yield" in df_dist.columns else np.nan

y_last = df_dist.loc[df_dist["Year"] == LAST_YEAR, "Yield"]
y_last = float(y_last.iloc[0]) if len(y_last) else np.nan

y_pred = df_dist.loc[df_dist["Year"] == CURR_YEAR, "Predicted_2025_Yield"]
y_pred = float(y_pred.iloc[0]) if len(y_pred) else np.nan

st.markdown(f"## 🔍 {sel_district}" + (f" — {sel_state}" if sel_state != "All" else ""))

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Mean Yield ({HIST_START}-{HIST_END})", "-" if pd.isna(mean_yield) else f"{mean_yield:.1f}")
k2.metric(f"Yield {LAST_YEAR}", "-" if pd.isna(y_last) else f"{y_last:.1f}")
k3.metric(f"Predicted Yield {CURR_YEAR}", "-" if pd.isna(y_pred) else f"{y_pred:.1f}")

if pd.notna(y_pred) and pd.notna(mean_yield):
    k4.metric("Δ 2025 vs Hist mean", f"{(y_pred - mean_yield):+.1f}")
elif pd.notna(y_pred) and pd.notna(y_last):
    k4.metric("Δ 2025 vs Last year", f"{(y_pred - y_last):+.1f}")
else:
    k4.metric("Δ", "-")


# -----------------------------
# COMPACT PLOTS BELOW
# -----------------------------
st.markdown("---")
st.subheader("📊 District Drivers (compact monthly profiles)")

# Row 1: Yield series (small)
cA, cB, cC = st.columns(3)
with cA:
    plot_yearly_small(df_dist, "Yield", "Observed Yield (Yearly)", "Yield")
with cB:
    plot_yearly_small(df_dist, "Predicted_2025_Yield", f"Predicted Yield (Yearly)", "Pred Yield")
with cC:
    # mini table
    cols_show = [c for c in ["Year", "Yield", "Predicted_2025_Yield"] if c in df_dist.columns]
    st.dataframe(df_dist[cols_show], use_container_width=True, height=240)

# Row 2/3: Weather + phenology in compact grid
g1, g2, g3 = st.columns(3)
with g1:
    plot_monthly_small(df_dist, "PRCP", "Rainfall (Jun–Nov)", "mm")
    plot_monthly_small(df_dist, "TMAX", "TMAX (Jun–Nov)", "°C")
with g2:
    plot_monthly_small(df_dist, "TMIN", "TMIN (Jun–Nov)", "°C")
    plot_monthly_small(df_dist, "RH", "RH (Jun–Nov)", "%")
with g3:
    plot_monthly_small(df_dist, "LAImax", "LAImax (Jun–Nov)", "LAI")
    plot_monthly_small(df_dist, "GPPsum", "GPPsum (Jun–Nov)", "GPP")

# Optional extra drivers
with st.expander("More drivers (optional)"):
    h1, h2, h3 = st.columns(3)
    with h1:
        plot_monthly_small(df_dist, "SRAD", "SRAD (Jun–Nov)", "unit")
    with h2:
        plot_monthly_small(df_dist, "WIND", "WIND (Jun–Nov)", "unit")
    with h3:
        st.info("Add any extra monthly variables here if your CSV has them (e.g., VPD, SoilMoist, etc.).")

st.markdown("---")
st.subheader("⬇️ Download selected district rows")
st.download_button(
    "Download CSV (selected district)",
    data=df_dist.to_csv(index=False).encode("utf-8"),
    file_name=f"soy_{sel_state}_{sel_district}_rows.csv".replace(" ", "_"),
    mime="text/csv"
)
