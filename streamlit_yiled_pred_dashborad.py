# streamlit_soy_acreage_plus_yield_map.py
# ------------------------------------------------------------
# STARTING POINT: Your Acreage Stability dashboard (reference code you shared)
# UPDATE: Add Yield Prediction layers + Weather/Phenology plots on district click
#
# What it does:
# 1) Keeps your Acreage Stability map + filters + district KPI panel
# 2) Adds a second map layer (toggle) colored by Predicted 2025 Yield
# 3) On district click, shows small monthly charts:
#    PRCP, TMAX, TMIN, LAI, GPP (Hist mean vs 2024 vs 2025)
# 4) Uses your yield features CSV + optional predictions CSV for 2025
#
# Requirements:
# pip install streamlit pandas geopandas matplotlib folium streamlit-folium branca
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

# ------------------------------------------------
# 0. BASIC CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Soybean Acreage + Yield Dashboard", layout="wide")
st.title("🌾 Soybean Acreage Stability + Yield Forecast Dashboard")

st.markdown(
    """
    This dashboard summarises **district-wise soybean acreage stability** and adds a
    **Yield Forecast (2025) + Weather & Phenology drivers** for the selected district.
    """
)

st.markdown(
    """
    <style>
    p, li { font-size: 1.05rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 1. PATH SETTINGS
# ------------------------------------------------
DEFAULT_ACREAGE_CSV = r"District_Acreage_Variation_R2_2025_f.csv"
DEFAULT_SHP         = r"3states.shp"

# yield master + predictions
DEFAULT_YIELD_FEATURES = r"soy_yield_features_new_rs.csv"
DEFAULT_YIELD_PRED     = r"soy_yield_predictions_2025.csv"  # optional

st.sidebar.header("🔧 Data Inputs")

acreage_csv_path = st.sidebar.text_input("Acreage stability CSV path:", DEFAULT_ACREAGE_CSV)
shp_path         = st.sidebar.text_input("Shapefile path (districts):", DEFAULT_SHP)

yield_features_csv = st.sidebar.text_input("Yield features CSV path:", DEFAULT_YIELD_FEATURES)
yield_pred_csv     = st.sidebar.text_input("Yield predictions CSV (optional):", DEFAULT_YIELD_PRED)

# Column names in your files (change if different)
district_col_csv = "District"
state_col_csv    = "State"

district_col_shp = "District"
state_col_shp    = "State"

# Optional: column for last year's acreage (2024)
acreage_2024_col = "Acreage_2024"   # change if needed

# Year settings
st.sidebar.header("📅 Year Settings (Yield)")
HIST_START = st.sidebar.number_input("Hist start year", value=2021, step=1)
HIST_END   = st.sidebar.number_input("Hist end year", value=2024, step=1)
LAST_YEAR  = st.sidebar.number_input("Last year", value=2024, step=1)
CURR_YEAR  = st.sidebar.number_input("Current year", value=2025, step=1)

# Plot sizing
st.sidebar.header("📉 Plot Size")
PLOT_W = st.sidebar.slider("Plot width", 3.6, 6.8, 4.6, 0.1)
PLOT_H = st.sidebar.slider("Plot height", 2.0, 4.2, 2.6, 0.1)

# Months (your yield file has Jun..Nov)
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]

# ------------------------------------------------
# 2. HELPERS
# ------------------------------------------------
def norm(s: pd.Series) -> pd.Series:
    return (s.astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[^A-Z0-9 ]", "", regex=True))

def to_num(df_: pd.DataFrame, cols):
    for c in cols:
        if c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")

def plot_monthly_small(df_dist: pd.DataFrame, prefix: str, title: str, ylabel: str):
    cols = [f"{prefix}_{m}" for m in MONTHS if f"{prefix}_{m}" in df_dist.columns]
    if not cols:
        st.info(f"Missing monthly columns: {prefix}_Jun..{prefix}_Nov")
        return

    to_num(df_dist, cols)

    hist = df_dist[(df_dist["Year"] >= HIST_START) & (df_dist["Year"] <= HIST_END)]
    last = df_dist[df_dist["Year"] == LAST_YEAR]
    curr = df_dist[df_dist["Year"] == CURR_YEAR]

    hist_mean = hist[cols].mean(axis=0, skipna=True) if not hist.empty else pd.Series(index=cols, dtype=float)
    last_val  = last[cols].iloc[0] if len(last) else None
    curr_val  = curr[cols].iloc[0] if len(curr) else None

    x = [c.split("_")[-1] for c in cols]

    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.plot(x, hist_mean.values, marker="o", linewidth=1.5, label=f"Hist mean ({HIST_START}-{HIST_END})")
    if last_val is not None:
        ax.plot(x, last_val.values, marker="o", linewidth=1.5, label=str(LAST_YEAR))
    if curr_val is not None:
        ax.plot(x, curr_val.values, marker="o", linewidth=1.5, label=str(CURR_YEAR))

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    st.pyplot(fig, use_container_width=True)

# ------------------------------------------------
# 3. LOAD DATA (ACREAGE + SHAPE)
# ------------------------------------------------
@st.cache_data
def load_acreage_table(path):
    df = pd.read_csv(path)

    if district_col_csv not in df.columns:
        raise ValueError(f"Column '{district_col_csv}' not found in acreage CSV")
    if "Acreage_Stability_Class" not in df.columns:
        raise ValueError("Column 'Acreage_Stability_Class' not found in acreage CSV")
    if "Predicted_2025_Acreage" not in df.columns:
        raise ValueError("Column 'Predicted_2025_Acreage' not found in acreage CSV")

    df["District_key"] = norm(df[district_col_csv])
    if state_col_csv in df.columns:
        df[state_col_csv] = df[state_col_csv].astype(str).str.strip()
        df["State_key"] = norm(df[state_col_csv])
    else:
        df["State_key"] = "NA"
    return df

@st.cache_data
def load_shapefile(path):
    gdf = gpd.read_file(path)

    if district_col_shp not in gdf.columns:
        raise ValueError(f"Column '{district_col_shp}' not found in shapefile")

    gdf["District"] = gdf[district_col_shp].astype(str).str.strip()
    gdf["District_key"] = norm(gdf["District"])

    if state_col_shp in gdf.columns:
        gdf["State"] = gdf[state_col_shp].astype(str).str.strip()
        gdf["State_key"] = norm(gdf["State"])
    else:
        gdf["State"] = "NA"
        gdf["State_key"] = "NA"

    return gdf

try:
    acre_df = load_acreage_table(acreage_csv_path)
    gdf = load_shapefile(shp_path)
except Exception as e:
    st.error(f"Error loading acreage or shapefile: {e}")
    st.stop()

# Join acreage -> geometry
gdf_join = gdf.merge(acre_df, on=["District_key"], how="left", suffixes=("", "_acre"))

# If State is missing in acreage table, use shapefile's
if state_col_csv not in acre_df.columns and "State" in gdf_join.columns:
    gdf_join[state_col_csv] = gdf_join["State"]

# ------------------------------------------------
# 4. LOAD YIELD FEATURES + (OPTIONAL) PREDICTIONS
# ------------------------------------------------
@st.cache_data
def load_yield_features(path):
    ydf = pd.read_csv(path)
    for c in ["State", "District", "Year"]:
        if c not in ydf.columns:
            raise ValueError(f"Missing '{c}' in yield features CSV")

    ydf["Year"] = pd.to_numeric(ydf["Year"], errors="coerce")
    if "Yield" in ydf.columns:
        ydf["Yield"] = pd.to_numeric(ydf["Yield"], errors="coerce")

    ydf["State"] = ydf["State"].astype(str).str.strip()
    ydf["District"] = ydf["District"].astype(str).str.strip()
    ydf["State_key"] = norm(ydf["State"])
    ydf["District_key"] = norm(ydf["District"])
    return ydf

@st.cache_data
def load_yield_predictions(path, curr_year):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["State_key","District_key","Year","Predicted_2025_Yield"])

    pdf = pd.read_csv(path)
    for c in ["State","District","Year"]:
        if c not in pdf.columns:
            raise ValueError(f"Missing '{c}' in yield predictions CSV")

    pdf["Year"] = pd.to_numeric(pdf["Year"], errors="coerce")
    pdf["State"] = pdf["State"].astype(str).str.strip()
    pdf["District"] = pdf["District"].astype(str).str.strip()
    pdf["State_key"] = norm(pdf["State"])
    pdf["District_key"] = norm(pdf["District"])

    # your model usually saved "Predicted_Yield"
    if "Predicted_Yield" in pdf.columns:
        ycol = "Predicted_Yield"
    elif "Predicted_2025_Yield" in pdf.columns:
        ycol = "Predicted_2025_Yield"
    else:
        cand = [c for c in pdf.columns if "pred" in c.lower() and "yield" in c.lower()]
        ycol = cand[0] if cand else None

    if ycol is None:
        return pd.DataFrame(columns=["State_key","District_key","Year","Predicted_2025_Yield"])

    out = pdf.loc[pdf["Year"] == curr_year, ["State_key","District_key","Year", ycol]].copy()
    out = out.rename(columns={ycol: "Predicted_2025_Yield"})
    out["Predicted_2025_Yield"] = pd.to_numeric(out["Predicted_2025_Yield"], errors="coerce")
    return out

try:
    ydf = load_yield_features(yield_features_csv)
    ypred = load_yield_predictions(yield_pred_csv, CURR_YEAR)
except Exception as e:
    st.error(f"Error loading yield files: {e}")
    st.stop()

# Merge predictions into yield features (only 2025 gets it)
ydf = ydf.merge(ypred, on=["State_key","District_key","Year"], how="left")

# Also attach predicted yield into map gdf (2025 only)
y_map = ydf.loc[ydf["Year"] == CURR_YEAR, ["State_key","District_key","Predicted_2025_Yield"]].drop_duplicates()
gdf_join = gdf_join.merge(y_map, on=["State_key","District_key"], how="left")

# ------------------------------------------------
# 5. SIDEBAR FILTERS (Acreage filters remain)
# ------------------------------------------------
st.sidebar.header("📌 Filters")

if state_col_csv in gdf_join.columns:
    state_list = sorted(gdf_join[state_col_csv].dropna().unique())
else:
    state_list = ["All"]

selected_state_filter = st.sidebar.selectbox("Select State:", options=["All"] + state_list, index=0)

stability_classes = [
    "Stable Acreage",
    "Moderately Variable",
    "Highly Volatile / Crop Switching Likely",
    "Marginal Acreage (Statistically Unstable)"
]
stab_opts = st.sidebar.multiselect(
    "Filter by Stability Class:",
    options=stability_classes,
    default=stability_classes
)

# Map theme selector
st.sidebar.header("🗺️ Map Theme")
map_theme = st.sidebar.radio(
    "Color districts by:",
    options=["Acreage Stability Class", f"Predicted Yield {CURR_YEAR}"],
    index=0
)

# ------------------------------------------------
# 6. APPLY FILTERS
# ------------------------------------------------
df_view = gdf_join.copy()

if selected_state_filter != "All" and state_col_csv in df_view.columns:
    df_view = df_view[df_view[state_col_csv] == selected_state_filter]

if stab_opts:
    df_view = df_view[df_view["Acreage_Stability_Class"].isin(stab_opts)]

if df_view.empty:
    st.warning("No districts match the selected filters.")
    st.stop()

# ------------------------------------------------
# 7. COLOR MAPPING
# ------------------------------------------------
def classify_color(stab_class: str) -> str:
    if pd.isna(stab_class):
        return "#CCCCCC"
    if "Marginal Acreage" in stab_class:
        return "#FF0000"
    if "Highly Volatile" in stab_class:
        return "#FF7F00"
    if "Moderately Variable" in stab_class:
        return "#FFFF00"
    if "Stable Acreage" in stab_class:
        return "#00A000"
    return "#CCCCCC"

# yield colormap (continuous)
yield_vals = df_view["Predicted_2025_Yield"].dropna()
if len(yield_vals) > 0:
    y_min, y_max = float(yield_vals.min()), float(yield_vals.max())
    if y_min == y_max:
        y_max = y_min + 1e-6
    yield_cmap = cm.linear.YlGnBu_09.scale(y_min, y_max)
    yield_cmap.caption = f"Predicted Yield ({CURR_YEAR})"
else:
    yield_cmap = None

# ------------------------------------------------
# 8. SUMMARY KPIs (Acreage)
# ------------------------------------------------
total_pred_2025 = df_view["Predicted_2025_Acreage"].sum(skipna=True)

if "Mean_Acreage" in df_view.columns:
    base_area = df_view["Mean_Acreage"].sum(skipna=True)
    delta_pct = ((total_pred_2025 - base_area) / base_area * 100) if base_area > 0 else None
else:
    base_area = None
    delta_pct = None

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Predicted 2025 Acreage (Lakh ha)", f"{total_pred_2025:.2f}")
with col2:
    if base_area is not None and delta_pct is not None:
        st.metric("Total Mean Acreage (Historical)", f"{base_area:.2f}")
with col3:
    if delta_pct is not None:
        st.metric("Δ 2025 vs Mean (%)", f"{delta_pct:.1f}%")

# ------------------------------------------------
# 9. MAP (LEFT) + DISTRICT DETAIL (RIGHT)
# ------------------------------------------------
st.subheader("🗺️ Map & District Insight (Click a district)")

map_col, detail_col = st.columns([1.4, 1])

with map_col:
    bounds = df_view.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="cartodbpositron")

    def style_fn(feature):
        if map_theme == "Acreage Stability Class":
            stab = feature["properties"].get("Acreage_Stability_Class")
            color = classify_color(stab)
        else:
            v = feature["properties"].get("Predicted_2025_Yield")
            if v is None or (isinstance(v, float) and np.isnan(v)) or (yield_cmap is None):
                color = "#CCCCCC"
            else:
                color = yield_cmap(v)
        return {"fillColor": color, "color": "black", "weight": 0.5, "fillOpacity": 0.75}

    tooltip_fields = [district_col_csv, state_col_csv, "Acreage_Stability_Class"]
    tooltip_alias  = ["District", "State", "Acreage Stability"]
    if "Predicted_2025_Yield" in df_view.columns:
        tooltip_fields += ["Predicted_2025_Yield"]
        tooltip_alias  += [f"Pred Yield {CURR_YEAR}"]

    gj = folium.GeoJson(
        data=df_view.to_json(),
        style_function=style_fn,
        highlight_function=lambda x: {"weight": 3, "color": "blue"},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_alias, sticky=False),
    )
    gj.add_to(m)

    if map_theme != "Acreage Stability Class" and yield_cmap is not None:
        yield_cmap.add_to(m)

    map_data = st_folium(m, width="100%", height=550, key="soy_map")

# Determine selected district from click
all_districts = sorted(df_view[district_col_csv].dropna().unique())
clicked_district = None
if map_data:
    clicked_props = None
    if map_data.get("last_active_drawing"):
        clicked_props = map_data["last_active_drawing"].get("properties", {})
    elif map_data.get("last_object_clicked"):
        clicked_props = map_data["last_object_clicked"].get("properties", {})
    if clicked_props:
        clicked_district = clicked_props.get(district_col_csv)

if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = all_districts[0]

if clicked_district:
    st.session_state["selected_district"] = clicked_district

selected_district = st.session_state["selected_district"]
if selected_district not in all_districts:
    selected_district = all_districts[0]
    st.session_state["selected_district"] = selected_district

drow = df_view[df_view[district_col_csv] == selected_district].iloc[0]

# ------------------------------------------------
# 10. DISTRICT DETAIL (Acreage) + Yield KPIs
# ------------------------------------------------
with detail_col:
    st.markdown(f"### 🔍 {selected_district}")

    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"**State:** {drow.get(state_col_csv, 'NA')}")
        st.markdown(f"**Stability Class:** {drow.get('Acreage_Stability_Class', 'NA')}")
        if "CV(%)" in drow and not pd.isna(drow["CV(%)"]):
            st.markdown(f"**Acreage CV:** {drow['CV(%)']:.2f}%")
        if "Trend_Slope" in drow and not pd.isna(drow["Trend_Slope"]):
            st.markdown(
                f"**Acreage Trend Slope:** {drow['Trend_Slope']:.4f} Lakh ha / year "
                f"({'⬆️ increasing' if drow['Trend_Slope']>0 else '⬇️ decreasing' if drow['Trend_Slope']<0 else 'flat'})"
            )

    with colB:
        mean_ac = drow["Mean_Acreage"] if "Mean_Acreage" in drow else None
        pred25 = drow["Predicted_2025_Acreage"] if "Predicted_2025_Acreage" in drow else np.nan

        st.markdown("**Acreage (Lakh ha)**")
        st.write(f"- Historical Mean: **{mean_ac:.3f}**" if mean_ac is not None and not pd.isna(mean_ac) else "- Historical Mean: NA")
        st.write(f"- Predicted 2025: **{pred25:.3f}**" if not pd.isna(pred25) else "- Predicted 2025: NA")

        has_2024 = acreage_2024_col in drow.index and not pd.isna(drow.get(acreage_2024_col, np.nan))
        if has_2024:
            ac2024 = drow[acreage_2024_col]
            st.write(f"- 2024 Acreage: **{ac2024:.3f}**")
            if ac2024 > 0:
                st.write(f"- Δ 2025 vs 2024: **{((pred25-ac2024)/ac2024*100):+.1f}%**")

# ------------------------------------------------
# 11. YIELD DRIVER SECTION (Below map)
# ------------------------------------------------
st.markdown("---")
st.subheader("🌦️ Yield Drivers (Weather + Phenology) — Selected District")

# Pull district time-series from yield features table using normalized key
dkey = norm(pd.Series([selected_district])).iloc[0]
skey = norm(pd.Series([drow.get(state_col_csv, "NA")])).iloc[0]

df_dist = ydf[(ydf["District_key"] == dkey) & (ydf["State_key"] == skey)].copy().sort_values("Year")

if df_dist.empty:
    st.warning("No yield-feature rows found for this district/state in your yield_features CSV.")
else:
    # Yield KPIs (Hist mean, last year, 2025 predicted)
    hist = df_dist[(df_dist["Year"] >= HIST_START) & (df_dist["Year"] <= HIST_END)]
    mean_y = hist["Yield"].mean(skipna=True) if "Yield" in df_dist.columns and not hist.empty else np.nan

    y_last = df_dist.loc[df_dist["Year"] == LAST_YEAR, "Yield"]
    y_last = float(y_last.iloc[0]) if len(y_last) else np.nan

    y_pred = df_dist.loc[df_dist["Year"] == CURR_YEAR, "Predicted_2025_Yield"]
    y_pred = float(y_pred.iloc[0]) if len(y_pred) else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Mean Yield ({HIST_START}-{HIST_END})", "-" if np.isnan(mean_y) else f"{mean_y:.1f}")
    k2.metric(f"Yield {LAST_YEAR}", "-" if np.isnan(y_last) else f"{y_last:.1f}")
    k3.metric(f"Predicted Yield {CURR_YEAR}", "-" if np.isnan(y_pred) else f"{y_pred:.1f}")
    k4.metric("Δ 2025 vs 2024", "-" if (np.isnan(y_pred) or np.isnan(y_last)) else f"{(y_pred-y_last):+.1f}")

    st.markdown("### Monthly comparison (Hist mean vs 2024 vs 2025)")

    c1, c2, c3 = st.columns(3)
    with c1:
        plot_monthly_small(df_dist, "PRCP", "Rainfall", "mm")
        plot_monthly_small(df_dist, "TMAX", "TMAX", "°C")
    with c2:
        plot_monthly_small(df_dist, "TMIN", "TMIN", "°C")
        if any([f"RH_{m}" in df_dist.columns for m in MONTHS]):
            plot_monthly_small(df_dist, "RH", "Relative Humidity", "%")
        else:
            st.caption("RH columns not found.")
    with c3:
        if any([f"LAImax_{m}" in df_dist.columns for m in MONTHS]):
            plot_monthly_small(df_dist, "LAImax", "LAImax", "LAI")
        else:
            st.caption("LAImax columns not found.")
        if any([f"GPPsum_{m}" in df_dist.columns for m in MONTHS]):
            plot_monthly_small(df_dist, "GPPsum", "GPPsum", "GPP")
        else:
            st.caption("GPPsum columns not found.")

    st.markdown("### Data table (selected district)")
    show_cols = [c for c in ["Year","Acerage","Yield","Predicted_2025_Yield"] if c in df_dist.columns]
    st.dataframe(df_dist[show_cols].sort_values("Year"), use_container_width=True)

# ------------------------------------------------
# 12. DISTRICT TABLE (Filtered)
# ------------------------------------------------
st.markdown("---")
st.subheader("📋 District-wise Metrics (Filtered districts)")

cols_to_show = [
    district_col_csv,
    "Acreage_Stability_Class",
    "Years_Available" if "Years_Available" in df_view.columns else None,
    "Mean_Acreage" if "Mean_Acreage" in df_view.columns else None,
    "Std_Acreage" if "Std_Acreage" in df_view.columns else None,
    "CV(%)" if "CV(%)" in df_view.columns else None,
    "Trend_Slope" if "Trend_Slope" in df_view.columns else None,
    "R2" if "R2" in df_view.columns else None,
    acreage_2024_col if acreage_2024_col in df_view.columns else None,
    "Predicted_2025_Acreage",
    "Predicted_2025_Yield" if "Predicted_2025_Yield" in df_view.columns else None
]
cols_to_show = [c for c in cols_to_show if c is not None and c in df_view.columns]

st.dataframe(df_view[cols_to_show].copy().sort_values(district_col_csv), use_container_width=True)
