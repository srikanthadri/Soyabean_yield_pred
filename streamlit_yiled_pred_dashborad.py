import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Soybean Yield + Weather + Phenology", layout="wide")
st.title("🌾 Soybean Yield Forecast & Driver Dashboard")
st.caption("District-wise yield + monthly weather & phenology comparisons (Hist mean vs 2024 vs 2025).")

# -----------------------------
# INPUTS
# -----------------------------
DEFAULT_DATA = r"soy_yield_features_new_rs.csv"
DEFAULT_PRED = r"soy_yield_predictions_2025.csv"  # optional

st.sidebar.header("📁 Data Inputs")
data_path = st.sidebar.text_input("Master features CSV:", DEFAULT_DATA)
pred_path = st.sidebar.text_input("2025 predictions CSV (optional):", DEFAULT_PRED)

st.sidebar.header("📅 Year Settings")
HIST_START = st.sidebar.number_input("Hist start year", value=2021, step=1)
HIST_END   = st.sidebar.number_input("Hist end year", value=2024, step=1)
LAST_YEAR  = st.sidebar.number_input("Last year", value=2024, step=1)
CURR_YEAR  = st.sidebar.number_input("Current year", value=2025, step=1)

MONTHS = ["Jun","Jul","Aug","Sep","Oct","Nov"]
MONTH_ORDER = {m:i for i,m in enumerate(MONTHS)}

# -----------------------------
# HELPERS
# -----------------------------
def norm(s: pd.Series) -> pd.Series:
    return (s.astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[^A-Z0-9 ]", "", regex=True))

@st.cache_data
def load_data(data_csv: str, pred_csv: str):
    df = pd.read_csv(data_csv)
    df["Year"] = pd.to_numeric(df.get("Year"), errors="coerce")
    if "Yield" in df.columns:
        df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

    # keys
    df["State"] = df["State"].astype(str).str.strip()
    df["District"] = df["District"].astype(str).str.strip()
    df["State_key"] = norm(df["State"])
    df["District_key"] = norm(df["District"])

    # merge predictions if available
    if pred_csv and Path(pred_csv).exists():
        pred = pd.read_csv(pred_csv)
        pred["Year"] = pd.to_numeric(pred.get("Year"), errors="coerce")
        pred["State"] = pred["State"].astype(str).str.strip()
        pred["District"] = pred["District"].astype(str).str.strip()
        pred["State_key"] = norm(pred["State"])
        pred["District_key"] = norm(pred["District"])

        # your file likely has Predicted_Yield
        pred_col = "Predicted_Yield" if "Predicted_Yield" in pred.columns else None
        if pred_col is None:
            # fallback: try common names
            for c in ["Predicted_2025_Yield", "PredictedYield", "y_pred"]:
                if c in pred.columns:
                    pred_col = c
                    break

        if pred_col:
            pred_small = pred.loc[pred["Year"] == CURR_YEAR, ["State_key","District_key","Year", pred_col]].copy()
            pred_small = pred_small.rename(columns={pred_col: "Predicted_2025_Yield"})
            df = df.merge(pred_small, on=["State_key","District_key","Year"], how="left")

    # if predictions already inside master
    if "Predicted_Yield" in df.columns and "Predicted_2025_Yield" not in df.columns:
        df["Predicted_2025_Yield"] = df["Predicted_Yield"]

    return df

def plot_monthly_comparison(df_dist, var_prefix, title, ylabel):
    # expects columns like PRCP_Jun ... PRCP_Nov
    cols = [f"{var_prefix}_{m}" for m in MONTHS if f"{var_prefix}_{m}" in df_dist.columns]
    if not cols:
        st.info(f"No monthly columns found for {var_prefix}_Jun..Nov")
        return

    # Build series for Hist mean, Last year, Current year
    hist = df_dist[(df_dist["Year"] >= HIST_START) & (df_dist["Year"] <= HIST_END)]
    last = df_dist[df_dist["Year"] == LAST_YEAR]
    curr = df_dist[df_dist["Year"] == CURR_YEAR]

    hist_mean = hist[cols].mean(axis=0, skipna=True) if not hist.empty else pd.Series(index=cols, dtype=float)
    last_val  = last[cols].iloc[0] if len(last) else pd.Series(index=cols, dtype=float)
    curr_val  = curr[cols].iloc[0] if len(curr) else pd.Series(index=cols, dtype=float)

    x = [c.split("_")[-1] for c in cols]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(x, hist_mean.values, marker="o", label=f"Hist mean ({HIST_START}-{HIST_END})")
    if len(last):
        ax.plot(x, last_val.values, marker="o", label=str(LAST_YEAR))
    if len(curr):
        ax.plot(x, curr_val.values, marker="o", label=str(CURR_YEAR))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

def plot_yearly_series(df_dist, ycol, title, ylabel):
    if ycol not in df_dist.columns:
        st.info(f"Missing column: {ycol}")
        return

    d = df_dist[["Year", ycol]].dropna().sort_values("Year")
    if d.empty:
        st.info(f"No data to plot for {ycol}")
        return

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(d["Year"].values, d[ycol].values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# -----------------------------
# LOAD
# -----------------------------
try:
    df = load_data(data_path, pred_path)
except Exception as e:
    st.error(f"Error reading input files: {e}")
    st.stop()

# -----------------------------
# FILTERS
# -----------------------------
states = sorted(df["State"].dropna().unique())
sel_state = st.sidebar.selectbox("Select State", ["All"] + states, index=0)

df_view = df.copy()
if sel_state != "All":
    df_view = df_view[df_view["State"] == sel_state]

districts = sorted(df_view["District"].dropna().unique())
sel_district = st.sidebar.selectbox("Select District", districts, index=0)

df_dist = df_view[df_view["District"] == sel_district].copy()
if df_dist.empty:
    st.warning("No rows for selected district.")
    st.stop()

# -----------------------------
# HEADER KPI BLOCK
# -----------------------------
st.markdown(f"## 🔎 {sel_district} ({sel_state if sel_state!='All' else df_dist['State'].iloc[0]})")

hist = df_dist[(df_dist["Year"] >= HIST_START) & (df_dist["Year"] <= HIST_END)]
mean_yield = hist["Yield"].mean(skipna=True) if "Yield" in df_dist.columns else np.nan

y_2024 = df_dist.loc[df_dist["Year"] == LAST_YEAR, "Yield"]
y_2024 = y_2024.iloc[0] if len(y_2024) else np.nan

pred_2025 = df_dist.loc[df_dist["Year"] == CURR_YEAR, "Predicted_2025_Yield"]
pred_2025 = pred_2025.iloc[0] if len(pred_2025) else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Mean Yield ({HIST_START}-{HIST_END})", "-" if pd.isna(mean_yield) else f"{mean_yield:.1f}")
c2.metric(f"Yield {LAST_YEAR}", "-" if pd.isna(y_2024) else f"{y_2024:.1f}")
c3.metric(f"Predicted Yield {CURR_YEAR}", "-" if pd.isna(pred_2025) else f"{pred_2025:.1f}")
if pd.notna(pred_2025) and pd.notna(mean_yield):
    c4.metric("Δ 2025 vs Hist mean", f"{(pred_2025-mean_yield):+.1f}")
else:
    c4.metric("Δ 2025 vs Hist mean", "-")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Yield", "🌦️ Weather (Monthly)", "🌱 Phenology (Monthly)"])

with tab1:
    left, right = st.columns([1, 1])

    with left:
        plot_yearly_series(df_dist, "Yield", "Observed Yield (Yearly)", "Yield (kg/ha or t/ha)")
        if "Predicted_2025_Yield" in df_dist.columns and df_dist["Predicted_2025_Yield"].notna().any():
            plot_yearly_series(df_dist, "Predicted_2025_Yield", "Predicted Yield (Yearly view if available)", "Predicted Yield")

    with right:
        # show small table for years
        cols_show = [c for c in ["Year","Acerage","Yield","Predicted_2025_Yield"] if c in df_dist.columns]
        st.dataframe(df_dist[cols_show].sort_values("Year"), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        plot_monthly_comparison(df_dist, "PRCP", "Rainfall (Jun–Nov) Monthly Profile", "mm")
        plot_monthly_comparison(df_dist, "RH", "Relative Humidity (Jun–Nov) Monthly Profile", "%")
        plot_monthly_comparison(df_dist, "SRAD", "Solar Radiation (Jun–Nov) Monthly Profile", "W/m² (or unit)")
    with c2:
        plot_monthly_comparison(df_dist, "TMAX", "TMAX (Jun–Nov) Monthly Profile", "°C")
        plot_monthly_comparison(df_dist, "TMIN", "TMIN (Jun–Nov) Monthly Profile", "°C")
        plot_monthly_comparison(df_dist, "WIND", "Wind Speed (Jun–Nov) Monthly Profile", "m/s")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        plot_monthly_comparison(df_dist, "LAImax", "LAImax (Jun–Nov) Monthly Profile", "LAI")
    with c2:
        plot_monthly_comparison(df_dist, "GPPsum", "GPPsum (Jun–Nov) Monthly Profile", "GPP (unit)")

    st.markdown("### Driver summary (quick)")
    # Seasonal quick features from monthly (display only)
    months = ["Jun","Jul","Aug","Sep","Oct","Nov"]
    def season_sum(prefix):
        cols = [f"{prefix}_{m}" for m in months if f"{prefix}_{m}" in df_dist.columns]
        if not cols: return np.nan, np.nan, np.nan
        hist = df_dist[(df_dist["Year"] >= HIST_START) & (df_dist["Year"] <= HIST_END)]
        curr = df_dist[df_dist["Year"] == CURR_YEAR]
        last = df_dist[df_dist["Year"] == LAST_YEAR]
        h = hist[cols].sum(axis=1, skipna=True).mean() if not hist.empty else np.nan
        c = curr[cols].sum(axis=1, skipna=True).iloc[0] if len(curr) else np.nan
        l = last[cols].sum(axis=1, skipna=True).iloc[0] if len(last) else np.nan
        return h, l, c

    pr_hist, pr_last, pr_curr = season_sum("PRCP")
    gpp_hist, gpp_last, gpp_curr = season_sum("GPPsum")

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Season PRCP (Hist mean)", "-" if pd.isna(pr_hist) else f"{pr_hist:.1f}")
    cc2.metric(f"Season PRCP ({LAST_YEAR})", "-" if pd.isna(pr_last) else f"{pr_last:.1f}")
    cc3.metric(f"Season PRCP ({CURR_YEAR})", "-" if pd.isna(pr_curr) else f"{pr_curr:.1f}")

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Season GPP (Hist mean)", "-" if pd.isna(gpp_hist) else f"{gpp_hist:.1f}")
    cc2.metric(f"Season GPP ({LAST_YEAR})", "-" if pd.isna(gpp_last) else f"{gpp_last:.1f}")
    cc3.metric(f"Season GPP ({CURR_YEAR})", "-" if pd.isna(gpp_curr) else f"{gpp_curr:.1f}")

st.markdown("---")
st.subheader("📌 Download district data (filtered)")
st.download_button(
    "Download selected district rows as CSV",
    data=df_dist.to_csv(index=False).encode("utf-8"),
    file_name=f"soy_{sel_state}_{sel_district}_rows.csv".replace(" ", "_"),
    mime="text/csv"
)

