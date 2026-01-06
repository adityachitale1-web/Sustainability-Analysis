import os
import io
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DATA_DIR = "data"

st.set_page_config(page_title="Green Energy Sustainability Dashboard", layout="wide")


# -----------------------------
# Download helpers
# -----------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_zip_from_dfs(dfs: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, df in dfs.items():
            zf.writestr(filename, df.to_csv(index=False))
    return buf.getvalue()


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


def format_number(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def safe_div(a, b):
    return np.where(b == 0, 0.0, a / b)


# -----------------------------
# App start
# -----------------------------
maybe_generate_data()
plants, gen, grid, weather, ef = load_data()

st.title("Green Energy – Sustainability Analyst Dashboard")
st.caption("Generation, curtailment, downtime, revenue, avoided CO₂, plus weather & grid drivers.")

# -----------------------------
# Downloads (top of page)
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
# Sidebar: filters
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
    plants_list = sorted(gen["plant_id"].dropna().unique())

    tech_sel = st.multiselect("Technology", techs, default=techs)
    region_sel = st.multiselect("Region", regions, default=regions)
    plant_sel = st.multiselect("Plant", plants_list, default=plants_list)

    st.markdown("---")
    st.header("Avoided CO₂ baseline")
    baseline = st.selectbox("Baseline", ["Grid_Average", "Coal", "Gas"], index=0)

# -----------------------------
# Filter
# -----------------------------
mask = (
    (gen["date"].dt.date >= start) &
    (gen["date"].dt.date <= end) &
    (gen["technology"].isin(tech_sel)) &
    (gen["region"].isin(region_sel)) &
    (gen["plant_id"].isin(plant_sel))
)
dff = gen.loc[mask].copy()

# Derived fields
dff["month"] = dff["date"].dt.to_period("M").astype(str)
dff["day"] = dff["date"].dt.date
dff["dow"] = dff["date"].dt.day_name()
dff["dow_num"] = dff["date"].dt.weekday
dff["availability"] = (1 - dff["downtime_hours"] / 24.0).clip(0, 1)
dff["capacity_factor"] = safe_div(dff["net_generation_mwh"], dff["capacity_mw"] * 24.0)

# Merge grid + compute avoided CO2
dff = dff.merge(grid, on="date", how="left")

if baseline == "Grid_Average":
    intensity_kg_per_kwh = dff["grid_intensity_gco2_per_kwh"] / 1000.0
else:
    factor = float(ef.loc[ef["baseline"] == baseline, "emission_factor_kgco2_per_kwh"].iloc[0])
    intensity_kg_per_kwh = factor

dff["avoided_co2_tonnes"] = (dff["net_generation_mwh"] * 1000.0 * intensity_kg_per_kwh) / 1000.0

# KPIs
total_net_mwh = float(dff["net_generation_mwh"].sum())
total_gross_mwh = float(dff["gross_generation_mwh"].sum())
total_curt_mwh = float(dff["curtailment_mwh"].sum())
curt_rate = (total_curt_mwh / total_gross_mwh) if total_gross_mwh > 0 else 0.0
total_revenue = float(dff["revenue"].sum())
total_avoided = float(dff["avoided_co2_tonnes"].sum())
avg_downtime = float(dff["downtime_hours"].mean()) if len(dff) else 0.0
avg_cf = float(dff["capacity_factor"].mean()) if len(dff) else 0.0
avg_avail = float(dff["availability"].mean()) if len(dff) else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Net Generation (MWh)", format_number(total_net_mwh))
k2.metric("Curtailment (MWh)", format_number(total_curt_mwh))
k3.metric("Curtailment Rate", f"{curt_rate:.2%}")
k4.metric("Revenue", f"{total_revenue:,.0f}")
k5.metric("Avoided CO₂ (tonnes)", format_number(total_avoided))
k6.metric("Avg Capacity Factor", f"{avg_cf:.2%}")

k7, k8 = st.columns(2)
k7.metric("Avg Availability", f"{avg_avail:.2%}")
k8.metric("Avg Downtime (hrs/day)", f"{avg_downtime:.2f}")

st.markdown("---")

# -----------------------------
# 1) Trends + cumulative
# -----------------------------
st.subheader("Trends")

daily = dff.groupby("date", as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    revenue=("revenue", "sum"),
    avoided_co2_tonnes=("avoided_co2_tonnes", "sum"),
    avg_cf=("capacity_factor", "mean"),
    avg_availability=("availability", "mean"),
    grid_intensity_gco2_per_kwh=("grid_intensity_gco2_per_kwh", "mean")
)
daily["curtailment_rate"] = safe_div(daily["curtailment_mwh"], daily["gross_generation_mwh"])
daily = daily.sort_values("date")
daily["cum_net_mwh"] = daily["net_generation_mwh"].cumsum()
daily["cum_avoided_t"] = daily["avoided_co2_tonnes"].cumsum()

t1, t2 = st.columns(2)
with t1:
    fig = px.line(daily, x="date", y="net_generation_mwh", title="Net Generation (Daily)")
    st.plotly_chart(fig, use_container_width=True)
with t2:
    fig = px.line(daily, x="date", y="cum_net_mwh", title="Cumulative Net Generation (YTD)")
    st.plotly_chart(fig, use_container_width=True)

t3, t4 = st.columns(2)
with t3:
    fig = px.line(daily, x="date", y="avoided_co2_tonnes", title=f"Avoided CO₂ (Daily) – {baseline}")
    st.plotly_chart(fig, use_container_width=True)
with t4:
    fig = px.line(daily, x="date", y="cum_avoided_t", title=f"Cumulative Avoided CO₂ (YTD) – {baseline}")
    st.plotly_chart(fig, use_container_width=True)

t5, t6 = st.columns(2)
with t5:
    fig = px.line(daily, x="date", y="curtailment_rate", title="Curtailment Rate (Daily)")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
with t6:
    fig = px.line(daily, x="date", y="avg_cf", title="Avg Capacity Factor (Daily)")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# 2) Mix and monthly view
# -----------------------------
st.subheader("Mix & Monthly Composition")

mix = dff.groupby("technology", as_index=False)["net_generation_mwh"].sum()
fig = px.pie(mix, names="technology", values="net_generation_mwh", title="Generation Mix by Technology")
st.plotly_chart(fig, use_container_width=True)

monthly_mix = dff.groupby(["month", "technology"], as_index=False)["net_generation_mwh"].sum()
fig = px.bar(
    monthly_mix.sort_values("month"),
    x="month",
    y="net_generation_mwh",
    color="technology",
    title="Monthly Net Generation by Technology (Stacked)",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# 3) Plant leaderboard + extra charts
# -----------------------------
st.subheader("Plant Performance (Ranking & Diagnostics)")

by_plant = dff.groupby(["plant_id", "technology", "region"], as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    revenue=("revenue", "sum"),
    avoided_co2_tonnes=("avoided_co2_tonnes", "sum"),
    avg_cf=("capacity_factor", "mean"),
    avg_availability=("availability", "mean"),
    avg_downtime=("downtime_hours", "mean"),
)
by_plant["curtailment_rate"] = safe_div(by_plant["curtailment_mwh"], by_plant["gross_generation_mwh"])

# Best/Worst cards
if len(by_plant):
    best_cf = by_plant.sort_values("avg_cf", ascending=False).iloc[0]
    worst_curt = by_plant.sort_values("curtailment_rate", ascending=False).iloc[0]
    b1, b2 = st.columns(2)
    b1.metric("Best Capacity Factor Plant", f"{best_cf['plant_id']} ({best_cf['technology']})", f"{best_cf['avg_cf']:.2%}")
    b2.metric("Highest Curtailment Plant", f"{worst_curt['plant_id']} ({worst_curt['technology']})", f"{worst_curt['curtailment_rate']:.2%}")

p1, p2 = st.columns(2)
with p1:
    fig = px.bar(
        by_plant.sort_values("net_generation_mwh", ascending=False),
        x="plant_id", y="net_generation_mwh", color="technology",
        title="Net Generation by Plant"
    )
    st.plotly_chart(fig, use_container_width=True)

with p2:
    fig = px.bar(
        by_plant.sort_values("avg_cf", ascending=False),
        x="plant_id", y="avg_cf", color="technology",
        title="Avg Capacity Factor by Plant"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

p3, p4 = st.columns(2)
with p3:
    fig = px.bar(
        by_plant.sort_values("curtailment_rate", ascending=False),
        x="plant_id", y="curtailment_rate", color="technology",
        title="Curtailment Rate by Plant"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with p4:
    fig = px.bar(
        by_plant.sort_values("avg_availability", ascending=False),
        x="plant_id", y="avg_availability", color="technology",
        title="Avg Availability by Plant"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(by_plant.sort_values("net_generation_mwh", ascending=False), use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------
# 4) Heatmaps (seasonality & weekly patterns)
# -----------------------------
st.subheader("Seasonality & Operating Patterns")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

hm_gen = dff.groupby(["month", "dow"], as_index=False)["net_generation_mwh"].sum()
hm_gen["dow"] = pd.Categorical(hm_gen["dow"], categories=weekday_order, ordered=True)
hm_gen_pivot = hm_gen.pivot(index="month", columns="dow", values="net_generation_mwh").fillna(0)

hm_curt = dff.groupby(["month", "dow"], as_index=False).agg(
    curtailment_mwh=("curtailment_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
)
hm_curt["dow"] = pd.Categorical(hm_curt["dow"], categories=weekday_order, ordered=True)
hm_curt["curtailment_rate"] = safe_div(hm_curt["curtailment_mwh"], hm_curt["gross_generation_mwh"])
hm_curt_pivot = hm_curt.pivot(index="month", columns="dow", values="curtailment_rate").fillna(0)

h1, h2 = st.columns(2)
with h1:
    fig = px.imshow(
        hm_gen_pivot,
        aspect="auto",
        title="Heatmap: Net Generation (Month × Day-of-week)",
        labels=dict(x="Day of Week", y="Month", color="MWh")
    )
    st.plotly_chart(fig, use_container_width=True)

with h2:
    fig = px.imshow(
        hm_curt_pivot,
        aspect="auto",
        title="Heatmap: Curtailment Rate (Month × Day-of-week)",
        labels=dict(x="Day of Week", y="Month", color="Rate")
    )
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# 5) Scatter diagnostics: economics + grid constraints
# -----------------------------
st.subheader("Diagnostics: Economics & Grid Constraints")

s1, s2 = st.columns(2)

with s1:
    fig = px.scatter(
        daily,
        x="net_generation_mwh",
        y="revenue",
        title="Revenue vs Net Generation (Daily)",
        hover_data=["date"]
    )
    st.plotly_chart(fig, use_container_width=True)

with s2:
    fig = px.scatter(
        daily,
        x="grid_intensity_gco2_per_kwh",
        y="curtailment_rate",
        title="Curtailment Rate vs Grid Emissions Intensity (Daily)",
        hover_data=["date"]
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# 6) Weather drivers (scatter, no trendline dependency)
# -----------------------------
st.subheader("Drivers (Weather vs Generation)")

dff_w = dff.merge(weather, on=["date", "region"], how="left")

driver_daily = dff_w.groupby(["date", "technology"], as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    solar_irradiance_kwh_m2=("solar_irradiance_kwh_m2", "mean"),
    wind_speed_m_s=("wind_speed_m_s", "mean"),
    rainfall_mm=("rainfall_mm", "mean"),
)

d1, d2 = st.columns(2)

with d1:
    tech_for_driver = st.selectbox("Select technology", sorted(driver_daily["technology"].unique()))
    dd = driver_daily[driver_daily["technology"] == tech_for_driver].copy()

    if tech_for_driver == "Solar":
        fig = px.scatter(dd, x="solar_irradiance_kwh_m2", y="net_generation_mwh",
                         title="Solar: Irradiance vs Net Generation")
        st.plotly_chart(fig, use_container_width=True)

        # simple PR proxy: generation per irradiance (not real PR, but useful signal)
        dd["mwh_per_irr"] = safe_div(dd["net_generation_mwh"], dd["solar_irradiance_kwh_m2"])
        fig2 = px.line(dd.sort_values("date"), x="date", y="mwh_per_irr", title="Solar PR Proxy: MWh per Irradiance")
        st.plotly_chart(fig2, use_container_width=True)

    elif tech_for_driver == "Wind":
        fig = px.scatter(dd, x="wind_speed_m_s", y="net_generation_mwh",
                         title="Wind: Wind Speed vs Net Generation")
        st.plotly_chart(fig, use_container_width=True)

    elif tech_for_driver == "Hydro":
        fig = px.scatter(dd, x="rainfall_mm", y="net_generation_mwh",
                         title="Hydro: Rainfall vs Net Generation")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Biomass
        fig = px.scatter(dd, x="rainfall_mm", y="net_generation_mwh",
                         title="Biomass: Rainfall vs Net Generation (weak relationship)")
        st.plotly_chart(fig, use_container_width=True)

with d2:
    st.markdown("**Interpretation tips**")
    st.write(
        "- Solar should generally increase with irradiance.\n"
        "- Wind should generally increase with wind speed.\n"
        "- Hydro may correlate with rainfall (lag not modeled).\n"
        "- High curtailment suggests grid constraints, not asset issues.\n"
        "- Availability and CF help separate operations vs resource variability."
    )

with st.expander("Data dictionary / assumptions"):
    st.write(
        """
**generation_daily.csv**
- gross_generation_mwh: generation after availability adjustment
- curtailment_mwh: energy not delivered due to grid constraints
- net_generation_mwh: delivered energy (gross - curtailment)
- downtime_hours: operational downtime
- price_per_mwh: simulated market price
- revenue: net_generation_mwh * price_per_mwh

**Derived in app**
- availability = 1 - downtime_hours/24
- capacity_factor = net_generation_mwh / (capacity_mw * 24)

**Avoided CO₂**
- Grid_Average uses daily grid intensity (gCO₂/kWh).
- Coal/Gas use fixed emission factors (kgCO₂/kWh).
"""
    )
