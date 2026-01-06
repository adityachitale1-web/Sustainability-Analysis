import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = "data"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Green Energy Sustainability Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
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
    Adds boolean column is_outlier_<col> based on IQR bounds.
    If group_cols provided, computes bounds per group (recommended: per technology).
    """
    out_col = f"is_outlier_{col}"
    df[out_col] = False

    if group_cols:
        for _, idx in df.groupby(group_cols).groups.items():
            lo, hi = iqr_bounds(df.loc[idx, col], k=k)
            if pd.notna(lo) and pd.notna(hi):
                df.loc[idx, out_col] = (df.loc[idx, col] < lo) | (df.loc[idx, col] > hi)
    else:
        lo, hi = iqr_bounds(df[col], k=k)
        if pd.notna(lo) and pd.notna(hi):
            df[out_col] = (df[col] < lo) | (df[col] > hi)

    return df


def format_number(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


# -----------------------------
# Data generation/loading
# -----------------------------
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


# -----------------------------
# Load
# -----------------------------
maybe_generate_data()
plants, gen, grid, weather, ef = load_data()

st.title("Green Energy – Sustainability Analyst Dashboard")
st.caption("Pie charts, distributions, heatmaps, and IQR-based outlier detection (no dataset download features).")

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
# Filter + feature engineering
# -----------------------------
mask = (
    (gen["date"].dt.date >= start) &
    (gen["date"].dt.date <= end) &
    (gen["technology"].isin(tech_sel)) &
    (gen["region"].isin(region_sel)) &
    (gen["plant_id"].isin(plant_sel))
)
dff = gen.loc[mask].copy()

dff["month"] = dff["date"].dt.to_period("M").astype(str)
dff["dow"] = pd.Categorical(dff["date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)

dff["availability"] = (1 - dff["downtime_hours"] / 24.0).clip(0, 1)
dff["capacity_factor"] = safe_div(dff["net_generation_mwh"], dff["capacity_mw"] * 24.0)
dff["curtailment_rate"] = safe_div(dff["curtailment_mwh"], dff["gross_generation_mwh"])

dff = dff.merge(grid, on="date", how="left")

if baseline == "Grid_Average":
    intensity_kg_per_kwh = dff["grid_intensity_gco2_per_kwh"] / 1000.0
else:
    factor = float(ef.loc[ef["baseline"] == baseline, "emission_factor_kgco2_per_kwh"].iloc[0])
    intensity_kg_per_kwh = factor

dff["avoided_co2_tonnes"] = (dff["net_generation_mwh"] * 1000.0 * intensity_kg_per_kwh) / 1000.0

# Outlier flags (per technology)
dff = add_outlier_flag(dff, "net_generation_mwh", group_cols=["technology"], k=outlier_k)
dff = add_outlier_flag(dff, "curtailment_rate", group_cols=["technology"], k=outlier_k)
dff = add_outlier_flag(dff, "downtime_hours", group_cols=["technology"], k=outlier_k)
dff = add_outlier_flag(dff, "capacity_factor", group_cols=["technology"], k=outlier_k)

dff["is_any_outlier"] = (
    dff["is_outlier_net_generation_mwh"] |
    dff["is_outlier_curtailment_rate"] |
    dff["is_outlier_downtime_hours"] |
    dff["is_outlier_capacity_factor"]
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
outlier_pct = float(dff["is_any_outlier"].mean()) if len(dff) else 0.0

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
k9.metric("Outlier %", f"{outlier_pct:.2%}")

st.markdown("---")

# -----------------------------
# Pie / Donut charts
# -----------------------------
st.subheader("Composition (Pie / Donut Charts)")

mix_gen = dff_agg.groupby("technology", as_index=False)["net_generation_mwh"].sum()
mix_rev = dff_agg.groupby("technology", as_index=False)["revenue"].sum()
mix_co2 = dff_agg.groupby("technology", as_index=False)["avoided_co2_tonnes"].sum()
mix_curt = dff_agg.groupby("technology", as_index=False)["curtailment_mwh"].sum()

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.plotly_chart(px.pie(mix_gen, names="technology", values="net_generation_mwh", title="Generation Mix"), use_container_width=True)
with p2:
    st.plotly_chart(px.pie(mix_rev, names="technology", values="revenue", title="Revenue Mix"), use_container_width=True)
with p3:
    st.plotly_chart(px.pie(mix_co2, names="technology", values="avoided_co2_tonnes", title=f"Avoided CO₂ Mix ({baseline})"), use_container_width=True)
with p4:
    st.plotly_chart(px.pie(mix_curt, names="technology", values="curtailment_mwh", title="Curtailment Share (Donut)", hole=0.45), use_container_width=True)

st.markdown("---")

# -----------------------------
# Distribution charts
# -----------------------------
st.subheader("Distributions (Histograms + Box Plots)")

d1, d2 = st.columns(2)
with d1:
    fig = px.histogram(dff, x="capacity_factor", color="technology", nbins=40, title="Capacity Factor Distribution (includes outliers)")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
with d2:
    fig = px.box(dff, x="technology", y="capacity_factor", points="outliers", title="Box Plot: Capacity Factor by Technology")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

d3, d4 = st.columns(2)
with d3:
    st.plotly_chart(px.histogram(dff, x="downtime_hours", color="technology", nbins=30, title="Downtime Hours Distribution (includes outliers)"),
                    use_container_width=True)
with d4:
    st.plotly_chart(px.box(dff, x="technology", y="downtime_hours", points="outliers", title="Box Plot: Downtime Hours by Technology"),
                    use_container_width=True)

d5, d6 = st.columns(2)
with d5:
    fig = px.histogram(dff, x="curtailment_rate", color="technology", nbins=40, title="Curtailment Rate Distribution (includes outliers)")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
with d6:
    fig = px.box(dff, x="technology", y="curtailment_rate", points="outliers", title="Box Plot: Curtailment Rate by Technology")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# Trends
# -----------------------------
st.subheader("Trends")

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
daily["net_7d_ma"] = daily["net_generation_mwh"].rolling(7, min_periods=1).mean()
daily["curt_7d_ma"] = daily["curtailment_rate"].rolling(7, min_periods=1).mean()

t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(px.line(daily, x="date", y=["net_generation_mwh", "net_7d_ma"], title="Net Generation (Daily + 7D MA)"),
                    use_container_width=True)
with t2:
    fig = px.line(daily, x="date", y=["curtailment_rate", "curt_7d_ma"], title="Curtailment Rate (Daily + 7D MA)")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

t3, t4 = st.columns(2)
with t3:
    st.plotly_chart(px.line(daily, x="date", y="revenue", title="Revenue (Daily)"), use_container_width=True)
with t4:
    st.plotly_chart(px.line(daily, x="date", y="avoided_co2_tonnes", title=f"Avoided CO₂ (Daily) – {baseline}"), use_container_width=True)

st.markdown("---")

# -----------------------------
# Heatmaps
# -----------------------------
st.subheader("Heatmaps")

hm_gen = dff_agg.groupby(["month", "dow"], as_index=False)["net_generation_mwh"].sum()
hm_gen_pivot = hm_gen.pivot(index="month", columns="dow", values="net_generation_mwh").fillna(0)

hm_curt = dff_agg.groupby(["month", "dow"], as_index=False).agg(
    curtailment_mwh=("curtailment_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
)
hm_curt["curtailment_rate"] = safe_div(hm_curt["curtailment_mwh"], hm_curt["gross_generation_mwh"])
hm_curt_pivot = hm_curt.pivot(index="month", columns="dow", values="curtailment_rate").fillna(0)

h1, h2 = st.columns(2)
with h1:
    fig = px.imshow(hm_gen_pivot, aspect="auto",
                    title="Heatmap: Net Generation (Month × Day-of-week)",
                    labels=dict(x="Day of Week", y="Month", color="MWh"))
    st.plotly_chart(fig, use_container_width=True)
with h2:
    fig = px.imshow(hm_curt_pivot, aspect="auto",
                    title="Heatmap: Curtailment Rate (Month × Day-of-week)",
                    labels=dict(x="Day of Week", y="Month", color="Rate"))
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

pm = dff_agg.groupby(["plant_id", "month"], as_index=False)["net_generation_mwh"].sum()
pm_pivot = pm.pivot(index="plant_id", columns="month", values="net_generation_mwh").fillna(0)
fig = px.imshow(pm_pivot, aspect="auto",
                title="Heatmap: Net Generation (Plant × Month)",
                labels=dict(x="Month", y="Plant", color="MWh"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# Outlier views
# -----------------------------
st.subheader("Outlier Diagnostics")

o1, o2 = st.columns(2)
with o1:
    fig = px.scatter(
        dff.assign(outlier=np.where(dff["is_any_outlier"], "Outlier", "Normal")),
        x="net_generation_mwh",
        y="revenue",
        color="outlier",
        hover_data=["date", "plant_id", "technology"],
        title="Net Generation vs Revenue (Outliers highlighted)",
    )
    st.plotly_chart(fig, use_container_width=True)

with o2:
    fig = px.scatter(
        dff.assign(outlier=np.where(dff["is_any_outlier"], "Outlier", "Normal")),
        x="grid_intensity_gco2_per_kwh",
        y="curtailment_rate",
        color="outlier",
        hover_data=["date", "plant_id", "technology"],
        title="Curtailment Rate vs Grid Intensity (Outliers highlighted)",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

outlier_rows = dff[dff["is_any_outlier"]].copy().sort_values(["date", "plant_id"]).head(200)
with st.expander("Outlier rows (first 200)"):
    show_cols = [
        "date", "plant_id", "technology", "region",
        "net_generation_mwh", "capacity_factor", "curtailment_rate",
        "downtime_hours", "grid_intensity_gco2_per_kwh",
        "is_outlier_net_generation_mwh", "is_outlier_capacity_factor",
        "is_outlier_curtailment_rate", "is_outlier_downtime_hours",
    ]
    st.dataframe(outlier_rows[show_cols], use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------
# Drivers
# -----------------------------
st.subheader("Drivers (Weather vs Generation)")

dff_w = dff_agg.merge(weather, on=["date", "region"], how="left")
driver_daily = dff_w.groupby(["date", "technology"], as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    solar_irradiance_kwh_m2=("solar_irradiance_kwh_m2", "mean"),
    wind_speed_m_s=("wind_speed_m_s", "mean"),
    rainfall_mm=("rainfall_mm", "mean"),
)

dc1, dc2 = st.columns(2)
with dc1:
    tech_for_driver = st.selectbox("Select technology for driver plot", sorted(driver_daily["technology"].unique()))
    dd = driver_daily[driver_daily["technology"] == tech_for_driver].copy()

    if tech_for_driver == "Solar":
        st.plotly_chart(px.scatter(dd, x="solar_irradiance_kwh_m2", y="net_generation_mwh",
                                   title="Solar: Irradiance vs Net Generation"),
                        use_container_width=True)
    elif tech_for_driver == "Wind":
        st.plotly_chart(px.scatter(dd, x="wind_speed_m_s", y="net_generation_mwh",
                                   title="Wind: Wind Speed vs Net Generation"),
                        use_container_width=True)
    elif tech_for_driver == "Hydro":
        st.plotly_chart(px.scatter(dd, x="rainfall_mm", y="net_generation_mwh",
                                   title="Hydro: Rainfall vs Net Generation"),
                        use_container_width=True)
    else:
        st.plotly_chart(px.scatter(dd, x="rainfall_mm", y="net_generation_mwh",
                                   title="Biomass: Rainfall vs Net Generation"),
                        use_container_width=True)

with dc2:
    st.markdown("**How to use this dashboard**")
    st.write(
        "- Pie charts: portfolio composition.\n"
        "- Distributions: variability and typical ranges.\n"
        "- Heatmaps: seasonality + weekday patterns.\n"
        "- Outliers: potential incidents (downtime) or grid constraints (curtailment)."
    )
