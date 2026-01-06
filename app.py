import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = "data"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Green Energy – Sustainability + ML Dashboard", layout="wide")


# -----------------------------
# Helpers (math / formatting)
# -----------------------------
def safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return np.where(b == 0, 0.0, a / b)


def format_number(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def iqr_bounds(s: pd.Series, k: float = 1.5):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 8:
        return (np.nan, np.nan)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def add_outlier_flag(df: pd.DataFrame, col: str, group_cols=None, k: float = 1.5):
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


def insight_box(title: str, bullets: list[str]):
    """Consistent business-insight callout under charts."""
    st.markdown(
        f"""
<div style="border:1px solid #e6e6e6; border-radius:10px; padding:12px; background:#fafafa">
<b>{title}</b>
<ul>
{''.join([f"<li>{b}</li>" for b in bullets])}
</ul>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# ML helpers (no sklearn/scipy)
# -----------------------------
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def train_logreg(X, y, lr=0.1, steps=2000, reg=1e-3):
    n, p = X.shape
    w = np.zeros(p, dtype="float64")
    b = 0.0

    for _ in range(int(steps)):
        z = X @ w + b
        p_hat = sigmoid(z)
        dw = (X.T @ (p_hat - y)) / n + reg * w
        db = np.mean(p_hat - y)
        w -= lr * dw
        b -= lr * db

    return w, b


def roc_curve_points(y_true, y_score):
    thresholds = np.unique(y_score)[::-1]
    tprs, fprs = [], []
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        TP = int(((y_pred == 1) & (y_true == 1)).sum())
        FP = int(((y_pred == 1) & (y_true == 0)).sum())
        TPR = TP / P if P else 0.0
        FPR = FP / N if N else 0.0
        tprs.append(TPR)
        fprs.append(FPR)

    return np.array(fprs, dtype="float64"), np.array(tprs, dtype="float64")


def auc_trapz(fpr, tpr):
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def confusion_matrix_counts(y_true, y_pred):
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TN, FP, FN, TP


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
# App start
# -----------------------------
maybe_generate_data()
plants, gen, grid, weather, ef = load_data()

st.title("Green Energy – Sustainability + ML Dashboard")
st.caption("Business insights are shown directly under each key visual for decision-making.")

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

# Merge grid intensity
dff = dff.merge(grid, on="date", how="left")

# Avoided CO2
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
# KPIs (top row)
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

# KPI insights
avg_price = float(dff_agg["price_per_mwh"].mean()) if len(dff_agg) else 0.0
estimated_lost_revenue = float(dff_agg["curtailment_mwh"].sum()) * avg_price
insight_box(
    "Business insight (KPIs)",
    [
        f"Estimated revenue lost due to curtailment (using avg price): {estimated_lost_revenue:,.0f}.",
        "Use the curtailment rate KPI to distinguish grid constraints vs plant performance.",
        "Use capacity factor + availability together to separate resource variability from operational reliability.",
    ],
)

st.markdown("---")

# -----------------------------
# Part-to-whole (each chart gets its own insight)
# -----------------------------
st.subheader("Part-to-Whole Relationships")

mix_gen = dff_agg.groupby("technology", as_index=False)["net_generation_mwh"].sum()
mix_rev = dff_agg.groupby("technology", as_index=False)["revenue"].sum()
mix_co2 = dff_agg.groupby("technology", as_index=False)["avoided_co2_tonnes"].sum()
mix_curt = dff_agg.groupby("technology", as_index=False)["curtailment_mwh"].sum()
mix_region_gen = dff_agg.groupby("region", as_index=False)["net_generation_mwh"].sum()
mix_region_curt = dff_agg.groupby("region", as_index=False)["curtailment_mwh"].sum()

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(px.pie(mix_gen, names="technology", values="net_generation_mwh", title="Generation Mix (Technology)"),
                    use_container_width=True)
    top = mix_gen.sort_values("net_generation_mwh", ascending=False).head(1)
    top_txt = f"{top.iloc[0]['technology']} ({top.iloc[0]['net_generation_mwh']/mix_gen['net_generation_mwh'].sum():.1%})" if len(top) else "N/A"
    insight_box("Business insight", [f"Generation is driven primarily by: {top_txt}.", "Concentration indicates where operational focus yields biggest impact."])

with c2:
    st.plotly_chart(px.pie(mix_rev, names="technology", values="revenue", title="Revenue Mix (Technology)"),
                    use_container_width=True)
    top = mix_rev.sort_values("revenue", ascending=False).head(1)
    top_txt = f"{top.iloc[0]['technology']} ({top.iloc[0]['revenue']/mix_rev['revenue'].sum():.1%})" if len(top) else "N/A"
    insight_box("Business insight", [f"Revenue is concentrated in: {top_txt}.", "Mismatch between revenue mix and generation mix suggests price or contract differences."])

with c3:
    st.plotly_chart(px.pie(mix_co2, names="technology", values="avoided_co2_tonnes", title=f"Avoided CO₂ Mix ({baseline})"),
                    use_container_width=True)
    top = mix_co2.sort_values("avoided_co2_tonnes", ascending=False).head(1)
    top_txt = f"{top.iloc[0]['technology']} ({top.iloc[0]['avoided_co2_tonnes']/mix_co2['avoided_co2_tonnes'].sum():.1%})" if len(top) else "N/A"
    insight_box("Business insight", [f"Most avoided CO₂ comes from: {top_txt}.", "Use this for ESG reporting and impact storytelling."])

c4, c5, c6 = st.columns(3)
with c4:
    st.plotly_chart(px.pie(mix_curt, names="technology", values="curtailment_mwh", title="Curtailment Share (Donut)", hole=0.45),
                    use_container_width=True)
    top = mix_curt.sort_values("curtailment_mwh", ascending=False).head(1)
    top_txt = f"{top.iloc[0]['technology']} ({top.iloc[0]['curtailment_mwh']/mix_curt['curtailment_mwh'].sum():.1%})" if len(top) else "N/A"
    insight_box("Business insight", [f"Curtailment is highest for: {top_txt}.", "Prioritize storage/grid solutions where curtailment is concentrated."])

with c5:
    st.plotly_chart(px.pie(mix_region_gen, names="region", values="net_generation_mwh", title="Generation Mix (Region)"),
                    use_container_width=True)
    top = mix_region_gen.sort_values("net_generation_mwh", ascending=False).head(1)
    top_txt = f"{top.iloc[0]['region']} ({top.iloc[0]['net_generation_mwh']/mix_region_gen['net_generation_mwh'].sum():.1%})" if len(top) else "N/A"
    insight_box("Business insight", [f"Generation is most concentrated in: {top_txt}.", "Regional concentration may create exposure to local grid constraints."])

with c6:
    st.plotly_chart(px.pie(mix_region_curt, names="region", values="curtailment_mwh", title="Curtailment Mix (Region)", hole=0.45),
                    use_container_width=True)
    top = mix_region_curt.sort_values("curtailment_mwh", ascending=False).head(1)
    top_txt = f"{top.iloc[0]['region']} ({top.iloc[0]['curtailment_mwh']/mix_region_curt['curtailment_mwh'].sum():.1%})" if len(top) else "N/A"
    insight_box("Business insight", [f"Grid constraints appear most severe in: {top_txt}.", "Use this to prioritize stakeholder engagement by region."])

st.markdown("---")

# -----------------------------
# Distributions (each chart gets its own insight)
# -----------------------------
st.subheader("Distributions")

d1, d2 = st.columns(2)
with d1:
    fig = px.histogram(dff, x="capacity_factor", color="technology", nbins=40, title="Capacity Factor Distribution")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Wide distributions imply higher forecasting uncertainty and variable output.", "Use for benchmarking assets within a technology."])

with d2:
    fig = px.histogram(dff, x="curtailment_rate", color="technology", nbins=40, title="Curtailment Rate Distribution")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["A heavy right tail indicates rare but severe constraint events.", "Mitigation options: storage, curtailment-aware dispatch, grid upgrades."])

d3, d4 = st.columns(2)
with d3:
    st.plotly_chart(px.histogram(dff, x="downtime_hours", color="technology", nbins=30, title="Downtime Hours Distribution"),
                    use_container_width=True)
    insight_box("Business insight", ["Downtime represents controllable loss (maintenance/reliability).", "Focus on recurring downtime patterns by plant."])

with d4:
    fig = px.box(dff, x="technology", y="capacity_factor", points="outliers", title="Box Plot: Capacity Factor by Technology")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Outliers here may indicate abnormal weather, plant issues, or data problems.", "Use IQR settings to tune sensitivity."])

st.markdown("---")

# -----------------------------
# Trends (each chart gets its own insight)
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
    insight_box("Business insight", ["Use the 7-day average to spot structural changes vs daily noise.", "Drops can indicate outages, resource dips, or increased curtailment."])

with t2:
    fig = px.line(daily, x="date", y=["curtailment_rate", "curt_7d_ma"], title="Curtailment Rate (Daily + 7D MA)")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Sustained increases suggest worsening grid constraints.", "Plan maintenance during high-curtailment periods to reduce opportunity cost."])

t3, t4 = st.columns(2)
with t3:
    st.plotly_chart(px.line(daily, x="date", y="revenue", title="Revenue (Daily)"), use_container_width=True)
    insight_box("Business insight", ["Revenue variability may reflect both energy and pricing changes.", "Investigate days where revenue doesn't track generation."])

with t4:
    st.plotly_chart(px.line(daily, x="date", y="avoided_co2_tonnes", title=f"Avoided CO₂ (Daily) – {baseline}"),
                    use_container_width=True)
    insight_box("Business insight", ["This is the primary ESG impact metric for the selected baseline.", "Use as input for periodic sustainability reporting."])

st.markdown("---")

# -----------------------------
# Heatmaps (each chart gets its own insight)
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
    insight_box("Business insight", ["Shows seasonality + weekly patterns.", "Use to schedule maintenance during lower-output windows."])

with h2:
    fig = px.imshow(hm_curt_pivot, aspect="auto",
                    title="Heatmap: Curtailment Rate (Month × Day-of-week)",
                    labels=dict(x="Day of Week", y="Month", color="Rate"))
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Highlights recurring constraint periods.", "Useful for storage/dispatch planning and grid stakeholder discussions."])

pm = dff_agg.groupby(["plant_id", "month"], as_index=False)["net_generation_mwh"].sum()
pm_pivot = pm.pivot(index="plant_id", columns="month", values="net_generation_mwh").fillna(0)
fig = px.imshow(pm_pivot, aspect="auto",
                title="Heatmap: Net Generation (Plant × Month)",
                labels=dict(x="Month", y="Plant", color="MWh"))
st.plotly_chart(fig, use_container_width=True)
insight_box("Business insight", ["Quickly identifies consistently underperforming plants.", "Use with risk scoring (below) to prioritize investigations."])

st.markdown("---")

# -----------------------------
# Outlier diagnostics (each chart gets insight)
# -----------------------------
st.subheader("Outlier Diagnostics")

o1, o2 = st.columns(2)
with o1:
    fig = px.scatter(
        dff.assign(outlier=np.where(dff["is_any_outlier"], "Outlier", "Normal")),
        x="net_generation_mwh",
        y="revenue",
        color="outlier",
        hover_data=["date", "plant_id", "technology", "region"],
        title="Net Generation vs Revenue (Outliers highlighted)",
    )
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Outliers here can indicate pricing anomalies, data issues, or unusual dispatch outcomes.", "Prioritize high-revenue anomalies for audit."])

with o2:
    fig = px.scatter(
        dff.assign(outlier=np.where(dff["is_any_outlier"], "Outlier", "Normal")),
        x="grid_intensity_gco2_per_kwh",
        y="curtailment_rate",
        color="outlier",
        hover_data=["date", "plant_id", "technology", "region"],
        title="Curtailment Rate vs Grid Intensity (Outliers highlighted)",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["High curtailment at typical grid intensity may indicate localized congestion.", "Use plant+region drilldown to guide grid engagement."])

risk = dff_agg.groupby(["plant_id", "technology", "region"], as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    avg_capacity_factor=("capacity_factor", "mean"),
    avg_availability=("availability", "mean"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
    avg_downtime=("downtime_hours", "mean"),
)
risk["curtailment_rate"] = np.where(risk["gross_generation_mwh"] > 0, risk["curtailment_mwh"] / risk["gross_generation_mwh"], 0.0)
risk["risk_score"] = 2.0 * risk["curtailment_rate"] + (1 - risk["avg_availability"]) + (1 - risk["avg_capacity_factor"])

with st.expander("Outlier rows (first 200)"):
    outlier_rows = dff[dff["is_any_outlier"]].copy().sort_values(["date", "plant_id"]).head(200)
    show_cols = [
        "date", "plant_id", "technology", "region",
        "net_generation_mwh", "capacity_factor", "curtailment_rate",
        "downtime_hours", "grid_intensity_gco2_per_kwh",
        "is_outlier_net_generation_mwh", "is_outlier_capacity_factor",
        "is_outlier_curtailment_rate", "is_outlier_downtime_hours",
    ]
    st.dataframe(outlier_rows[show_cols], use_container_width=True, hide_index=True)

st.markdown("### Plants to Review (highest operational + curtailment risk)")
st.dataframe(risk.sort_values("risk_score", ascending=False).head(10), use_container_width=True, hide_index=True)
insight_box("Business insight", ["This is an action list: start with the highest risk score plants.", "Risk combines curtailment, availability and capacity factor (tunable weights)."])

st.markdown("---")

# -----------------------------
# ML: ROC, Confusion Matrix, Feature Importance (each chart gets insight)
# -----------------------------
st.subheader("ML: Predict High Curtailment Events (ROC / Confusion Matrix / Feature Importance)")

model_df = dff_agg.merge(weather, on=["date", "region"], how="left").copy()
model_df["curtailment_rate"] = safe_div(model_df["curtailment_mwh"], model_df["gross_generation_mwh"])

threshold = st.slider("High curtailment event threshold (curtailment rate)", 0.02, 0.20, 0.08, 0.01)
model_df["high_curtailment_event"] = (model_df["curtailment_rate"] >= threshold).astype(int)

num_features = [
    "grid_intensity_gco2_per_kwh",
    "solar_irradiance_kwh_m2",
    "wind_speed_m_s",
    "rainfall_mm",
    "downtime_hours",
    "capacity_mw",
]
num_features = [c for c in num_features if c in model_df.columns]

X_num = model_df[num_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
X_cat = pd.get_dummies(model_df[["technology", "region"]], drop_first=True)
X = pd.concat([X_num, X_cat], axis=1)
y = model_df["high_curtailment_event"].values.astype(int)

if len(X) < 200 or int(y.sum()) < 10 or int((y == 0).sum()) < 10:
    st.warning("Not enough rows/events in the current filters to train/evaluate the classifier. Expand date range / plants / regions.")
else:
    test_pct = st.slider("Test size (%)", 10, 40, 25, 5) / 100.0
    rng = np.random.default_rng(7)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_pct))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train = X.values[train_idx]
    X_test = X.values[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train_s, mu, sigma = standardize(X_train)
    X_test_s = (X_test - mu) / sigma

    lr = st.slider("Learning rate", 0.01, 0.5, 0.10, 0.01)
    steps = st.slider("Training steps", 500, 5000, 2000, 500)
    reg = st.slider("L2 regularization", 0.0, 0.01, 0.001, 0.001)

    w, b = train_logreg(X_train_s, y_train, lr=lr, steps=steps, reg=reg)
    y_score = sigmoid(X_test_s @ w + b)

    fpr, tpr = roc_curve_points(y_test, y_score)
    auc = auc_trapz(fpr, tpr)

    pred_threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    y_pred = (y_score >= pred_threshold).astype(int)

    TN, FP, FN, TP = confusion_matrix_counts(y_test, y_pred)
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC", f"{auc:.3f}")
    m2.metric("Accuracy", f"{acc:.3f}")
    m3.metric("Precision", f"{precision:.3f}")
    m4.metric("Recall", f"{recall:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        fig = px.line(roc_df, x="FPR", y="TPR", title="ROC Curve")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        insight_box("Business insight", ["Higher AUC means better early-warning capability.", "Use threshold tuning to balance false alarms vs missed events."])

    with c2:
        cm = pd.DataFrame([[TN, FP], [FN, TP]], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        insight_box("Business insight", ["False negatives = missed high-curtailment days (risk).", "False positives = unnecessary interventions (cost)."])

    feat_names = X.columns.tolist()
    imp = np.abs(w)
    imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False).head(20)

    fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                 title="Feature Importance (|coefficient| on standardized features)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Top features indicate the strongest drivers of high curtailment in this data.", "Use to target monitoring signals and operational playbooks."])

    with st.expander("Model details"):
        st.write(
            """
This classifier is a logistic regression trained on the filtered dataset (after optional outlier exclusion).

Label: high_curtailment_event = 1 if curtailment_rate ≥ selected threshold.

Features:
- Grid emissions intensity,
- Weather (irradiance / wind speed / rainfall),
- Operations (downtime, capacity),
- Technology and region (one-hot encoded).

Feature importance shown as absolute standardized coefficients.
"""
        )
