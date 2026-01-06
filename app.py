import os
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = "data"

st.set_page_config(page_title="Green Energy Sustainability Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_zip_from_dfs(dfs: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, df in dfs.items():
            zf.writestr(filename, df.to_csv(index=False))
    return buf.getvalue()


def safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return np.where(b == 0, 0.0, a / b)


def iqr_bounds(s: pd.Series, k: float = 1.5):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 8:
        return (np.nan, np.nan)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def add_outlier_flag(df: pd.DataFrame, col: str, group_cols=None, k: float = 1.5):
    """
    Adds boolean column f"is_outlier_{col}" using IQR bounds.
    If group_cols provided, apply bounds per group (e.g., per technology).
    """
    out_col = f"is_outlier_{col}"
    df[out_col] = False

    if group_cols:
        for keys, idx in df.groupby(group_cols).groups.items():
            lo, hi = iqr_bounds(df.loc[idx, col], k=k)
            if pd.notna(lo) and pd.notna(hi):
                df.loc[idx, out_col] = (df.loc[idx, col] < lo) | (df.loc[idx, col] > hi)
    else:
        lo, hi = iqr_bounds(df[col], k=k)
        if pd.notna(lo) and pd.notna(hi):
            df[out_col] = (df[col] < lo) | (df[col] > hi)

    return df


@st.cache_data
def maybe_generate_data():
    required = [
        "plants.csv",
        "generation_daily.csv",
        "grid_intensity_daily.csv",
        "weather_daily.csv",
        "emissions_factors.csv",
    ]
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        import generate_datasets
        generate_datasets.generate()


@st.cache_data
def load_data():
    plants = pd.read_csv(os.path.join(DATA_DIR, "plants.csv"))
    gen = pd.read_csv(os.path.join(DATA_DIR, "generation_daily.csv"))
    grid = pd.read_csv(os.path.join(DATA_DIR, "grid_intensity_daily.csv"))
    weather = pd.read_csv(os.path.join(DATA_DIR, "weather_daily.csv"))
    ef = pd.read_csv(os.path.join(DATA_DIR, "emissions_factors.csv"))

    gen["date"] = pd.to_datetime(gen["date"], errors="coerce")
    grid["date"] = pd.to_datetime(grid["date"], errors="coerce")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

    gen = gen.dropna(subset=["date"])
    grid = grid.dropna(subset=["date"])
    weather = weather.dropna(subset=["date"])

    return plants, gen, grid, weather, ef


def format_number(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


# -----------------------------
# Load
# -----------------------------
maybe_generate_data()
plants, gen, grid, weather, ef = load_data()

st.title("Green Energy – Sustainability Analyst Dashboard")
st.caption("Includes heatmaps, outlier detection, operational diagnostics, and sustainability impact metrics.")

# -----------------------------
# Downloads (main page)
# -----------------------------
st.subheader("Download datasets (CSV)")

plants_dl = plants.copy()

gen_dl = gen.copy()
gen_dl["date"] = pd.to_datetime(gen_dl["date"]).dt.strftime("%Y-%m-%d")

grid_dl = grid.copy()
grid_dl["date"] = pd.to_datetime(grid_dl["date"]).dt.strftime("%Y-%m-%d")

weather_dl = weather.copy()
weather_dl["date"] = pd.to_datetime(weather_dl["date"]).dt.strftime("%Y-%m-%d")

ef_dl = ef.copy()

zip_bytes = make_zip_from_dfs({
    "plants.csv": plants_dl,
    "generation_daily.csv": gen_dl,
    "grid_intensity_daily.csv": grid_dl,
    "weather_daily.csv": weather_dl,
    "emissions_factors.csv": ef_dl,
})

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("Download plants.csv", df_to_csv_bytes(plants_dl), "plants.csv", "text/csv", use_container_width=True)
    st.download_button("Download emissions_factors.csv", df_to_csv_bytes(ef_dl), "emissions_factors.csv", "text/csv", use_container_width=True)
with c2:
    st.download_button("Download generation_daily.csv", df_to_csv_bytes(gen_dl), "generation_daily.csv", "text/csv", use_container_width=True)
    st.download_button("Download grid_intensity_daily.csv", df_to_csv_bytes(grid_dl), "grid_intensity_daily.csv", "text/csv", use_container_width=True)
with c3:
    st.download_button("Download weather_daily.csv", df_to_csv_bytes(weather_dl), "weather_daily.csv", "text/csv", use_container_width=True)
    st.download_button("Download ALL datasets (ZIP)", zip_bytes, "green_energy_datasets.zip", "application/zip", use_container_width=True)

st.markdown("---")

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("Filters")

    min_date = gen["date"].min().date()
    max_date = gen["date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_date, max_date

    techs = sorted(gen["technology"].dropna().unique())
    regions = sorted(gen["region"].dropna().unique())
    plant_ids = sorted(gen["plant_id"].dropna().unique())

    tech_sel = st.multiselect("Technology", techs, default=techs)
    region_sel = st.multiselect("Region", regions, default=regions)
    plant_sel = st.multiselect("Plant", plant_ids, default=plant_ids)

    st.markdown("---")
    st.header("Avoided CO₂ baseline")
    baseline = st.selectbox("Baseline", ["Grid_Average", "Coal", "Gas"], index=0)

    st.markdown("---")
    st.header("Outliers")
    outlier_k = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
    exclude_outliers = st.checkbox("Exclude outliers from aggregations", value=False)

# -----------------------------
# Filtered dataset
# -----------------------------
mask = (
    (gen["date"].dt.date >= start) &
    (gen["date"].dt.date <= end) &
    (gen["technology"].isin(tech_sel)) &
    (gen["region"].isin(region_sel)) &
    (gen["plant_id"].isin(plant_sel))
)
dff = gen.loc[mask].copy()

# Derived
dff["month"] = dff["date"].dt.to_period("M").astype(str)
dff["dow"] = dff["date"].dt.day_name()
dff["dow_num"] = dff["date"].dt.weekday
dff["availability"] = (1 - dff["downtime_hours"] / 24.0).clip(0, 1)
dff["capacity_factor"] = safe_div(dff["net_generation_mwh"], dff["capacity_mw"] * 24.0)
dff["curtailment_rate"] = safe_div(dff["curtailment_mwh"], dff["gross_generation_mwh"])

# Merge grid + avoided CO2
dff = dff.merge(grid, on="date", how="left")

if baseline == "Grid_Average":
    intensity_kg_per_kwh = dff["grid_intensity_gco2_per_kwh"] / 1000.0
else:
    factor = float(ef.loc[ef["baseline"] == baseline, "emission_factor_kgco2_per_kwh"].iloc[0])
    intensity_kg_per_kwh = factor

dff["avoided_co2_tonnes"] = (dff["net_generation_mwh"] * 1000.0 * intensity_kg_per_kwh) / 1000.0

# Outlier flags (per technology so Solar/Wind/Hydro aren't compared unfairly)
dff = add_outlier_flag(dff, "net_generation_mwh", group_cols=["technology"], k=outlier_k)
dff = add_outlier_flag(dff, "curtailment_rate", group_cols=["technology"], k=outlier_k)
dff = add_outlier_flag(dff, "downtime_hours", group_cols=["technology"], k=outlier_k)

dff["is_any_outlier"] = (
    dff["is_outlier_net_generation_mwh"] |
    dff["is_outlier_curtailment_rate"] |
    dff["is_outlier_downtime_hours"]
)

dff_agg = dff.loc[~dff["is_any_outlier"]].copy() if exclude_outliers else dff.copy()

# -----------------------------
# KPIs
# -----------------------------
total_net_mwh = float(dff_agg["net_generation_mwh"].sum())
total_gross_mwh = float(dff_agg["gross_generation_mwh"].sum())
total_curt_mwh = float(dff_agg["curtailment_mwh"].sum())
curt_rate = (total_curt_mwh / total_gross_mwh) if total_gross_mwh > 0 else 0.0
total_revenue = float(dff_agg["revenue"].sum())
total_avoided = float(dff_agg["avoided_co2_tonnes"].sum())
avg_cf = float(dff_agg["capacity_factor"].mean()) if len(dff_agg) else 0.0
avg_avail = float(dff_agg["availability"].mean()) if len(dff_agg) else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Net Generation (MWh)", format_number(total_net_mwh))
k2.metric("Curtailment (MWh)", format_number(total_curt_mwh))
k3.metric("Curtailment Rate", f"{curt_rate:.2%}")
k4.metric("Revenue", f"{total_revenue:,.0f}")
k5.metric("Avoided CO₂ (tonnes)", format_number(total_avoided))
k6.metric("Avg Capacity Factor", f"{avg_cf:.2%}")

k7, k8, k9 = st.columns(3)
k7.metric("Avg Availability", f"{avg_avail:.2%}")
k8.metric("Outlier Rows", f"{int(dff['is_any_outlier'].sum()):,}")
k9.metric("Outlier %", f"{(dff['is_any_outlier'].mean() if len(dff) else 0):.2%}")

st.markdown("---")

# -----------------------------
# Trends + rolling averages
# -----------------------------
st.subheader("Trends (with rolling averages)")

daily = dff_agg.groupby("date", as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    revenue=("revenue", "sum"),
    avoided_co2_tonnes=("avoided_co2_tonnes", "sum"),
    avg_cf=("capacity_factor", "mean"),
    avg_avail=("availability", "mean"),
    grid_intensity_gco2_per_kwh=("grid_intensity_gco2_per_kwh", "mean"),
)
daily["curtailment_rate"] = safe_div(daily["curtailment_mwh"], daily["gross_generation_mwh"])
daily = daily.sort_values("date")

for col in ["net_generation_mwh", "curtailment_rate", "avg_cf", "revenue"]:
    daily[f"{col}_7d_ma"] = daily[col].rolling(7, min_periods=1).mean()

t1, t2 = st.columns(2)
with t1:
    fig = px.line(daily, x="date", y=["net_generation_mwh", "net_generation_mwh_7d_ma"], title="Net Generation (Daily + 7D MA)")
    st.plotly_chart(fig, use_container_width=True)
with t2:
    fig = px.line(daily, x="date", y=["curtailment_rate", "curtailment_rate_7d_ma"], title="Curtailment Rate (Daily + 7D MA)")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

t3, t4 = st.columns(2)
with t3:
    fig = px.line(daily, x="date", y=["avg_cf", "avg_cf_7d_ma"], title="Capacity Factor (Avg Daily + 7D MA)")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
with t4:
    fig = px.line(daily, x="date", y=["revenue", "revenue_7d_ma"], title="Revenue (Daily + 7D MA)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# Heatmaps
# -----------------------------
st.subheader("Heatmaps")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

hm_gen = dff_agg.groupby(["month", "dow"], as_index=False)["net_generation_mwh"].sum()
hm_gen["dow"] = pd.Categorical(hm_gen["dow"], categories=weekday_order, ordered=True)
hm_gen_pivot = hm_gen.pivot(index="month", columns="dow", values="net_generation_mwh").fillna(0)

hm_curt = dff_agg.groupby(["month", "dow"], as_index=False).agg(
    curtailment_mwh=("curtailment_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
)
hm_curt["dow"] = pd.Categorical(hm_curt["dow"], categories=weekday_order, ordered=True)
hm_curt["curtailment_rate"] = safe_div(hm_curt["curtailment_mwh"], hm_curt["gross_generation_mwh"])
hm_curt_pivot = hm_curt.pivot(index="month", columns="dow", values="curtailment_rate").fillna(0)

h1, h2 = st.columns(2)
with h1:
    fig = px.imshow(
        hm_gen_pivot, aspect="auto",
        title="Heatmap: Net Generation (Month × Day-of-week)",
        labels=dict(x="Day of Week", y="Month", color="MWh")
    )
    st.plotly_chart(fig, use_container_width=True)

with h2:
    fig = px.imshow(
        hm_curt_pivot, aspect="auto",
        title="Heatmap: Curtailment Rate (Month × Day-of-week)",
        labels=dict(x="Day of Week", y="Month", color="Rate")
    )
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# Plant x Month heatmap
pm = dff_agg.groupby(["plant_id", "month"], as_index=False)["net_generation_mwh"].sum()
pm_pivot = pm.pivot(index="plant_id", columns="month", values="net_generation_mwh").fillna(0)
fig = px.imshow(
    pm_pivot, aspect="auto",
    title="Heatmap: Net Generation (Plant × Month)",
    labels=dict(x="Month", y="Plant", color="MWh")
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# Outlier visualizations
# -----------------------------
st.subheader("Outliers & Distributions")

o1, o2 = st.columns(2)
with o1:
    fig = px.box(dff, x="technology", y="capacity_factor", points="outliers", title="Box Plot: Capacity Factor by Technology")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with o2:
    fig = px.box(dff, x="technology", y="curtailment_rate", points="outliers", title="Box Plot: Curtailment Rate by Technology")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

o3, o4 = st.columns(2)
with o3:
    fig = px.histogram(dff, x="downtime_hours", nbins=30, color="technology", title="Downtime Hours Distribution")
    st.plotly_chart(fig, use_container_width=True)

with o4:
    fig = px.histogram(dff, x="capacity_factor", nbins=30, color="technology", title="Capacity Factor Distribution")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# Scatter highlighting outliers
s1, s2 = st.columns(2)
with s1:
    fig = px.scatter(
        dff.assign(outlier=np.where(dff["is_any_outlier"], "Outlier", "Normal")),
        x="net_generation_mwh", y="revenue", color="outlier",
        hover_data=["date", "plant_id", "technology"],
        title="Net Generation vs Revenue (Outliers highlighted)"
    )
    st.plotly_chart(fig, use_container_width=True)

with s2:
    fig = px.scatter(
        dff.assign(outlier=np.where(dff["is_any_outlier"], "Outlier", "Normal")),
        x="grid_intensity_gco2_per_kwh", y="curtailment_rate", color="outlier",
        hover_data=["date", "plant_id", "technology"],
        title="Curtailment Rate vs Grid Intensity (Outliers highlighted)"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# Outlier table
outlier_rows = dff[dff["is_any_outlier"]].copy()
outlier_rows = outlier_rows.sort_values(["date", "plant_id"]).head(200)

with st.expander("Outlier rows (first 200)"):
    show_cols = [
        "date", "plant_id", "technology", "region",
        "net_generation_mwh", "capacity_factor", "curtailment_rate",
        "downtime_hours", "grid_intensity_gco2_per_kwh",
        "is_outlier_net_generation_mwh", "is_outlier_curtailment_rate", "is_outlier_downtime_hours"
    ]
    for c in ["capacity_factor", "curtailment_rate"]:
        if c in outlier_rows.columns:
            # keep as decimals, but readable
            pass
    st.dataframe(outlier_rows[show_cols], use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------
# Top constraint days
# -----------------------------
st.subheader("Grid Constraint Days (Highest Curtailment Rate)")

daily_full = dff.groupby("date", as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    grid_intensity_gco2_per_kwh=("grid_intensity_gco2_per_kwh", "mean"),
)
daily_full["curtailment_rate"] = safe_div(daily_full["curtailment_mwh"], daily_full["gross_generation_mwh"])
top_constraints = daily_full.sort_values("curtailment_rate", ascending=False).head(15)

fig = px.bar(top_constraints, x="date", y="curtailment_rate", title="Top 15 Curtailment Rate Days")
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(top_constraints, use_container_width=True, hide_index=True)

# -----------------------------
# Notes
# -----------------------------
with st.expander("What outliers mean here"):
    st.write(
        """
Outliers are flagged using an IQR rule (per-technology), which helps catch:
- unusually low/high net generation for a given technology
- unusual curtailment rates (possible grid congestion events)
- unusual downtime spikes (possible operational incidents)

You can choose to exclude outliers from aggregated charts using the sidebar toggle.
"""
    )
