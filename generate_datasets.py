import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
RNG_SEED = 42

def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def clip_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Winsorize (cap) using IQR to avoid unrealistic outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return series.clip(lower=lo, upper=hi)

def generate():
    ensure_dir()
    rng = np.random.default_rng(RNG_SEED)

    # ---------- plants.csv ----------
    plants = pd.DataFrame([
        {"plant_id":"SOL-01","plant_name":"Sunridge Solar","technology":"Solar","region":"North","latitude":28.7,"longitude":77.1,"capacity_mw":120},
        {"plant_id":"SOL-02","plant_name":"DesertBloom Solar","technology":"Solar","region":"West","latitude":23.0,"longitude":72.6,"capacity_mw":180},
        {"plant_id":"WND-01","plant_name":"CoastalWind Farm","technology":"Wind","region":"South","latitude":13.1,"longitude":80.3,"capacity_mw":150},
        {"plant_id":"WND-02","plant_name":"Highland Wind","technology":"Wind","region":"East","latitude":22.6,"longitude":88.4,"capacity_mw":200},
        {"plant_id":"HYD-01","plant_name":"RiverRun Hydro","technology":"Hydro","region":"North","latitude":30.3,"longitude":78.0,"capacity_mw":250},
        {"plant_id":"BIO-01","plant_name":"AgriCycle Biomass","technology":"Biomass","region":"Central","latitude":21.1,"longitude":79.1,"capacity_mw":90},
    ])
    # basic validation
    plants["capacity_mw"] = plants["capacity_mw"].clip(lower=1)
    plants.to_csv(os.path.join(DATA_DIR, "plants.csv"), index=False)

    # ---------- date range ----------
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # ---------- grid_intensity_daily.csv ----------
    # gCO2/kWh – seasonal + noise, clipped to plausible range
    base = 620 + 60*np.sin(np.linspace(0, 2*np.pi, n)) + rng.normal(0, 25, n)
    grid_intensity = pd.DataFrame({
        "date": dates,
        "grid_intensity_gco2_per_kwh": np.clip(base, 420, 780)
    })
    grid_intensity.to_csv(os.path.join(DATA_DIR, "grid_intensity_daily.csv"), index=False)

    # ---------- weather_daily.csv ----------
    # Simple regional drivers (same day, region-specific noise)
    regions = plants["region"].unique().tolist()
    weather_rows = []
    for region in regions:
        # solar irradiance kWh/m2/day
        irr = 4.8 + 1.2*np.sin(np.linspace(-0.6, 2*np.pi-0.6, n)) + rng.normal(0, 0.4, n)
        irr = np.clip(irr, 2.0, 7.5)

        # wind speed m/s
        wind = 5.5 + 1.5*np.sin(np.linspace(0.2, 2*np.pi+0.2, n)) + rng.normal(0, 0.8, n)
        wind = np.clip(wind, 1.5, 12.0)

        # rainfall mm (many zeros)
        rain = rng.gamma(shape=1.2, scale=6.0, size=n)
        rain[rng.random(n) < 0.6] = 0
        rain = np.clip(rain, 0, 120)

        for d, i, w, r in zip(dates, irr, wind, rain):
            weather_rows.append({
                "date": d,
                "region": region,
                "solar_irradiance_kwh_m2": float(i),
                "wind_speed_m_s": float(w),
                "rainfall_mm": float(r)
            })
    weather = pd.DataFrame(weather_rows)
    weather.to_csv(os.path.join(DATA_DIR, "weather_daily.csv"), index=False)

    # ---------- emissions_factors.csv ----------
    # For avoided emissions comparisons
    ef = pd.DataFrame([
        {"baseline":"Coal","emission_factor_kgco2_per_kwh":0.95},
        {"baseline":"Gas","emission_factor_kgco2_per_kwh":0.45},
        {"baseline":"Grid_Average","emission_factor_kgco2_per_kwh":np.nan},  # will use grid_intensity_daily
    ])
    ef.to_csv(os.path.join(DATA_DIR, "emissions_factors.csv"), index=False)

    # ---------- generation_daily.csv ----------
    # Generate daily energy by plant driven by tech + weather + availability
    gen_rows = []
    for _, p in plants.iterrows():
        plant_id = p["plant_id"]
        tech = p["technology"]
        cap = float(p["capacity_mw"])
        region = p["region"]

        wreg = weather[weather["region"] == region].set_index("date")

        # capacity factor model
        if tech == "Solar":
            cf = (wreg["solar_irradiance_kwh_m2"] / 7.0) * 0.30 + rng.normal(0, 0.03, n)
            cf = np.clip(cf, 0.05, 0.32)
        elif tech == "Wind":
            cf = (wreg["wind_speed_m_s"] / 12.0) * 0.50 + rng.normal(0, 0.05, n)
            cf = np.clip(cf, 0.08, 0.55)
        elif tech == "Hydro":
            # hydro inversely affected by low rainfall (simplified)
            cf = 0.42 + 0.08*np.tanh((wreg["rainfall_mm"] - 10)/20) + rng.normal(0, 0.03, n)
            cf = np.clip(cf, 0.20, 0.55)
        else:  # Biomass
            cf = 0.70 + rng.normal(0, 0.04, n)
            cf = np.clip(cf, 0.50, 0.85)

        # Availability/downtime (hours out of 24)
        downtime_hours = rng.poisson(lam=0.4, size=n).astype(float)
        downtime_hours = np.clip(downtime_hours, 0, 10)

        availability = 1 - downtime_hours/24.0
        availability = np.clip(availability, 0.6, 1.0)

        # potential generation MWh = MW * 24 * cf * availability
        energy_mwh = cap * 24 * cf * availability

        # curtailment % (higher for solar/wind sometimes)
        curt_rate = rng.normal(0.03 if tech in ["Solar", "Wind"] else 0.01, 0.01, n)
        curt_rate = np.clip(curt_rate, 0.0, 0.12)
        curtailment_mwh = energy_mwh * curt_rate
        net_mwh = energy_mwh - curtailment_mwh

        # price (INR/MWh or USD/MWh — label neutral as "price_per_mwh")
        # seasonal mild variation
        price = 55 + 10*np.sin(np.linspace(0, 2*np.pi, n)) + rng.normal(0, 4, n)
        price = np.clip(price, 30, 90)

        revenue = net_mwh * price

        dfp = pd.DataFrame({
            "date": dates,
            "plant_id": plant_id,
            "region": region,
            "technology": tech,
            "capacity_mw": cap,
            "gross_generation_mwh": energy_mwh,
            "curtailment_mwh": curtailment_mwh,
            "net_generation_mwh": net_mwh,
            "downtime_hours": downtime_hours,
            "price_per_mwh": price,
            "revenue": revenue
        })

        # Clean numeric outliers via IQR caps (winsorize) to keep realistic
        for col in ["gross_generation_mwh", "curtailment_mwh", "net_generation_mwh", "revenue"]:
            dfp[col] = clip_iqr(dfp[col])

        # hard constraints
        dfp["gross_generation_mwh"] = dfp["gross_generation_mwh"].clip(lower=0)
        dfp["curtailment_mwh"] = dfp["curtailment_mwh"].clip(lower=0)
        dfp["net_generation_mwh"] = (dfp["gross_generation_mwh"] - dfp["curtailment_mwh"]).clip(lower=0)
        dfp["downtime_hours"] = dfp["downtime_hours"].clip(lower=0, upper=24)
        dfp["price_per_mwh"] = dfp["price_per_mwh"].clip(lower=0)
        dfp["revenue"] = (dfp["net_generation_mwh"] * dfp["price_per_mwh"]).clip(lower=0)

        gen_rows.append(dfp)

    generation = pd.concat(gen_rows, ignore_index=True)
    generation.to_csv(os.path.join(DATA_DIR, "generation_daily.csv"), index=False)

    print("Generated datasets in ./data/")
    print("Files:", os.listdir(DATA_DIR))

if __name__ == "__main__":
    generate()
