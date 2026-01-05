import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DATA_DIR = "data"

st.set_page_config(page_title="Green Energy Sustainability Dashboard", layout="wide")

@st.cache_data
def maybe_generate_data():
    # If data missing, generate it (uses generate_datasets.py)
    required = ["plants.csv", "generation_daily.csv", "grid_intensity_daily.csv", "weather_daily.csv", "emissions_factors.csv"]
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

    gen["date"] = pd.to_datetime(gen["date"])
    grid["date"] = pd.to_datetime(grid["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    return plants, gen, grid, weather, ef

def format_number(x):
    return f"{x:,.0f}"

maybe_generate_data()
plants, gen, grid, weather, ef = load_data()

st.title("Green Energy – Sustainability Analyst Dashboard")
st.caption("Synthetic but realistic datasets: generation, curtailment, downtime, revenue, avoided CO₂, and weather drivers.")

# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.header("Filters")

    min_date = gen["date"].min().date()
    max_date = gen["date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_date, max_date

    techs = sorted(gen["technology"].unique())
    regions = sorted(gen["region"].unique())
    plants_list = sorted(gen["plant_id"].unique())

    tech_sel = st.multiselect("Technology", techs, default=techs)
    region_sel = st.multiselect("Region", regions, default=regions)
    plant_sel = st.multiselect("Plant", plants_list, default=plants_list)

    st.markdown("---")
    st.header("Avoided CO₂ baseline")
    baseline = st.selectbox("Baseline", ["Grid_Average", "Coal", "Gas"], index=0)
    st.caption("Avoided CO₂ = Net Generation × (baseline intensity). Grid_Average uses daily grid intensity.")

# Filter
mask = (
    (gen["date"].dt.date >= start) &
    (gen["date"].dt.date <= end) &
    (gen["technology"].isin(tech_sel)) &
    (gen["region"].isin(region_sel)) &
    (gen["plant_id"].isin(plant_sel))
)
dff = gen.loc[mask].copy()

# Merge grid intensity for avoided emissions
dff = dff.merge(grid, on="date", how="left")

# baseline intensity in kgCO2/kWh
if baseline == "Grid_Average":
    # grid intensity is gCO2/kWh -> kgCO2/kWh
    intensity_kg_per_kwh = dff["grid_intensity_gco2_per_kwh"] / 1000.0
else:
    factor = float(ef.loc[ef["baseline"] == baseline, "emission_factor_kgco2_per_kwh"].iloc[0])
    intensity_kg_per_kwh = factor

# Avoided CO2: net_generation_mwh -> kwh = * 1000
dff["avoided_co2_tonnes"] = (dff["net_generation_mwh"] * 1000.0 * intensity_kg_per_kwh) / 1000.0  # kg -> tonnes

# KPIs
total_net_mwh = dff["net_generation_mwh"].sum()
total_gross_mwh = dff["gross_generation_mwh"].sum()
total_curt_mwh = dff["curtailment_mwh"].sum()
curt_rate = (total_curt_mwh / total_gross_mwh) if total_gross_mwh > 0 else 0
total_revenue = dff["revenue"].sum()
total_avoided = dff["avoided_co2_tonnes"].sum()
avg_downtime = dff["downtime_hours"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Net Generation (MWh)", format_number(total_net_mwh))
k2.metric("Curtailment (MWh)", format_number(total_curt_mwh))
k3.metric("Curtailment Rate", f"{curt_rate:.2%}")
k4.metric("Revenue", f"{total_revenue:,.0f}")
k5.metric("Avoided CO₂ (tonnes)", format_number(total_avoided))
k6.metric("Avg Downtime (hrs/day)", f"{avg_downtime:.2f}")

st.markdown("---")

# ----------------------------
# Trends
# ----------------------------
daily = dff.groupby("date", as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    revenue=("revenue", "sum"),
    avoided_co2_tonnes=("avoided_co2_tonnes", "sum"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
)
daily["curtailment_rate"] = np.where(daily["gross_generation_mwh"] > 0, daily["curtailment_mwh"]/daily["gross_generation_mwh"], 0)

t1, t2 = st.columns(2)
with t1:
    fig = px.line(daily, x="date", y="net_generation_mwh", title="Net Generation Trend (MWh)")
    st.plotly_chart(fig, use_container_width=True)
with t2:
    fig = px.line(daily, x="date", y="avoided_co2_tonnes", title=f"Avoided CO₂ Trend (tonnes) – Baseline: {baseline}")
    st.plotly_chart(fig, use_container_width=True)

t3, t4 = st.columns(2)
with t3:
    fig = px.line(daily, x="date", y="revenue", title="Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)
with t4:
    fig = px.line(daily, x="date", y="curtailment_rate", title="Curtailment Rate Trend")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------
# Plant / Tech performance
# ----------------------------
st.subheader("Operational & Sustainability Performance")

by_plant = dff.groupby(["plant_id", "technology", "region"], as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    downtime_hours=("downtime_hours", "mean"),
    avoided_co2_tonnes=("avoided_co2_tonnes", "sum"),
    revenue=("revenue", "sum"),
)
by_plant["curtailment_rate"] = np.where(by_plant["gross_generation_mwh"] > 0, by_plant["curtailment_mwh"]/by_plant["gross_generation_mwh"], 0)

p1, p2 = st.columns([1.2, 0.8])
with p1:
    fig = px.bar(
        by_plant.sort_values("net_generation_mwh", ascending=False),
        x="plant_id", y="net_generation_mwh", color="technology",
        title="Net Generation by Plant", barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

with p2:
    fig = px.bar(
        by_plant.sort_values("curtailment_rate", ascending=False),
        x="plant_id", y="curtailment_rate", color="technology",
        title="Curtailment Rate by Plant"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

p3, p4 = st.columns(2)
with p3:
    fig = px.bar(
        by_plant.sort_values("avoided_co2_tonnes", ascending=False),
        x="plant_id", y="avoided_co2_tonnes", color="technology",
        title="Avoided CO₂ by Plant (tonnes)"
    )
    st.plotly_chart(fig, use_container_width=True)

with p4:
    fig = px.bar(
        by_plant.sort_values("downtime_hours", ascending=False),
        x="plant_id", y="downtime_hours", color="technology",
        title="Avg Downtime (hrs/day) by Plant"
    )
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    by_plant.sort_values("net_generation_mwh", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ----------------------------
# Renewable mix
# ----------------------------
mix = dff.groupby("technology", as_index=False)["net_generation_mwh"].sum()
fig = px.pie(mix, names="technology", values="net_generation_mwh", title="Generation Mix by Technology")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------
# Drivers: weather vs generation
# ----------------------------
st.subheader("Drivers (Weather vs Generation)")

# Join weather by date+region, then aggregate
dff_w = dff.merge(weather, on=["date", "region"], how="left")

driver_daily = dff_w.groupby(["date", "technology"], as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    solar_irradiance_kwh_m2=("solar_irradiance_kwh_m2", "mean"),
    wind_speed_m_s=("wind_speed_m_s", "mean"),
    rainfall_mm=("rainfall_mm", "mean"),
)

d1, d2 = st.columns(2)
with d1:
    tech_for_driver = st.selectbox("Select technology for driver view", sorted(driver_daily["technology"].unique()))
    dd = driver_daily[driver_daily["technology"] == tech_for_driver].copy()

    if tech_for_driver == "Solar":
        fig = px.scatter(dd, x="solar_irradiance_kwh_m2", y="net_generation_mwh",
                         title="Solar: Irradiance vs Net Generation", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    elif tech_for_driver == "Wind":
        fig = px.scatter(dd, x="wind_speed_m_s", y="net_generation_mwh",
                         title="Wind: Wind Speed vs Net Generation", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    elif tech_for_driver == "Hydro":
        fig = px.scatter(dd, x="rainfall_mm", y="net_generation_mwh",
                         title="Hydro: Rainfall vs Net Generation", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.scatter(dd, x="rainfall_mm", y="net_generation_mwh",
                         title="Biomass: Rainfall vs Net Generation (weak relationship)", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

with d2:
    st.markdown("**Interpretation tips**")
    st.write(
        "- Solar should correlate with irradiance.\n"
        "- Wind should correlate with wind speed.\n"
        "- Hydro may correlate with rainfall (lag not modeled here).\n"
        "- Curtailment spikes often indicate grid constraints."
    )

with st.expander("Data dictionary / assumptions"):
    st.write("""
**generation_daily.csv**
- gross_generation_mwh: theoretical generation after availability
- curtailment_mwh: grid-constrained energy not delivered
- net_generation_mwh: delivered energy (gross - curtailment)
- downtime_hours: operational downtime
- price_per_mwh: simulated market price
- revenue: net_generation_mwh * price_per_mwh

**Avoided CO₂**
- Grid_Average uses daily grid intensity (gCO₂/kWh).
- Coal/Gas use fixed emission factors (kgCO₂/kWh).
""")
