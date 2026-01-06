import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = "data"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Green Energy – Sustainability + Finance + ML Dashboard", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stMarkdownContainer"] { line-height: 1.45; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
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
    st.markdown(
        f"""
<div style="
    border:1px solid #d0d0d0;
    border-radius:10px;
    padding:12px 14px;
    background:#ffffff;
    color:#111111;
    font-size:15px;
    line-height:1.45;
">
  <div style="color:#0b3d91; font-weight:700; margin-bottom:6px;">
    {title}
  </div>
  <ul style="margin:0; padding-left:18px; color:#111111;">
    {''.join([f"<li style='margin: 4px 0;'>{b}</li>" for b in bullets])}
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
        tprs.append(TP / P if P else 0.0)
        fprs.append(FP / N if N else 0.0)
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
# Finance derivations (deterministic, clean)
# -----------------------------
def add_financials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - opex: simulated operating expenditure based on tech, capacity, and downtime
    - profit: revenue - opex
    - margin: profit / revenue
    - cost_per_mwh: opex / net_generation_mwh
    - profit_per_mwh: profit / net_generation_mwh
    - energy_product + product_mwh + product_revenue: simplified product breakdown
    """
    out = df.copy()

    # OPEX rates (currency units per MW-day) by technology (illustrative)
    base_opex_per_mw_day = {
        "Solar": 35.0,
        "Wind": 45.0,
        "Hydro": 60.0,
        "Biomass": 90.0,
    }
    # variable cost per MWh (e.g., biomass fuel cost)
    var_cost_per_mwh = {
        "Solar": 0.5,
        "Wind": 0.8,
        "Hydro": 1.2,
        "Biomass": 18.0,
    }

    out["base_opex_per_mw_day"] = out["technology"].map(base_opex_per_mw_day).fillna(50.0)
    out["var_cost_per_mwh"] = out["technology"].map(var_cost_per_mwh).fillna(2.0)

    # Downtime penalty: more downtime -> higher maintenance cost
    out["downtime_multiplier"] = (1.0 + 0.03 * out["downtime_hours"]).clip(1.0, 2.0)

    # OPEX = fixed (capacity-based) + variable (generation-based)
    out["opex"] = (
        (out["capacity_mw"] * out["base_opex_per_mw_day"] * out["downtime_multiplier"])
        + (out["net_generation_mwh"] * out["var_cost_per_mwh"])
    ).clip(lower=0)

    out["profit"] = (out["revenue"] - out["opex"]).clip(lower=-1e12)
    out["margin"] = np.where(out["revenue"] > 0, out["profit"] / out["revenue"], 0.0)

    out["cost_per_mwh"] = np.where(out["net_generation_mwh"] > 0, out["opex"] / out["net_generation_mwh"], 0.0)
    out["profit_per_mwh"] = np.where(out["net_generation_mwh"] > 0, out["profit"] / out["net_generation_mwh"], 0.0)

    # Energy products (simple split)
    # - Day-ahead: majority baseline
    # - Peak: higher price premium, more on weekdays
    # - Green Credits: small % of volume, higher price premium
    dow = out["date"].dt.weekday  # Mon=0
    weekday = (dow <= 4).astype(float)

    out["share_peak"] = (0.18 + 0.07 * weekday).clip(0.10, 0.30)
    out["share_green"] = 0.05  # constant
    out["share_day_ahead"] = (1.0 - out["share_peak"] - out["share_green"]).clip(0.60, 0.85)

    # Build long form "product" table and merge back later as needed
    return out

def build_product_table(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    # MWh allocation
    tmp["mwh_day_ahead"] = tmp["net_generation_mwh"] * tmp["share_day_ahead"]
    tmp["mwh_peak"] = tmp["net_generation_mwh"] * tmp["share_peak"]
    tmp["mwh_green"] = tmp["net_generation_mwh"] * tmp["share_green"]

    # price multipliers
    tmp["price_day_ahead"] = tmp["price_per_mwh"] * 1.00
    tmp["price_peak"] = tmp["price_per_mwh"] * 1.20
    tmp["price_green"] = tmp["price_per_mwh"] * 1.35

    rows = []
    for prod, mwh_col, price_col in [
        ("Day-ahead", "mwh_day_ahead", "price_day_ahead"),
        ("Peak", "mwh_peak", "price_peak"),
        ("Green Credits", "mwh_green", "price_green"),
    ]:
        part = tmp[["date", "plant_id", "technology", "region", "capacity_mw", "downtime_hours"]].copy()
        part["energy_product"] = prod
        part["product_mwh"] = tmp[mwh_col].clip(lower=0)
        part["product_revenue"] = (tmp[mwh_col] * tmp[price_col]).clip(lower=0)
        rows.append(part)

    prod_df = pd.concat(rows, ignore_index=True)
    return prod_df

# -----------------------------
# App start
# -----------------------------
maybe_generate_data()
plants, gen, grid, weather, ef = load_data()

st.title("Green Energy – Sustainability + Finance + ML Dashboard")
st.caption("Adds income/expenditure/profit views and energy product mix alongside sustainability + ML diagnostics.")

# Sidebar filters
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

# Filter
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
dff["dow"] = pd.Categorical(dff["date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
dff["availability"] = (1 - dff["downtime_hours"] / 24.0).clip(0, 1)
dff["capacity_factor"] = safe_div(dff["net_generation_mwh"], dff["capacity_mw"] * 24.0)
dff["curtailment_rate"] = safe_div(dff["curtailment_mwh"], dff["gross_generation_mwh"])

# Grid + avoided CO2
dff = dff.merge(grid, on="date", how="left")
if baseline == "Grid_Average":
    intensity_kg_per_kwh = dff["grid_intensity_gco2_per_kwh"] / 1000.0
else:
    factor = float(ef.loc[ef["baseline"] == baseline, "emission_factor_kgco2_per_kwh"].iloc[0])
    intensity_kg_per_kwh = factor
dff["avoided_co2_tonnes"] = (dff["net_generation_mwh"] * 1000.0 * intensity_kg_per_kwh) / 1000.0

# Outliers
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

# Finance columns
dff_agg = add_financials(dff_agg)
dff_fin = dff_agg.copy()
product_df = build_product_table(dff_fin)

# -----------------------------
# KPIs
# -----------------------------
total_net_mwh = float(dff_agg["net_generation_mwh"].sum())
total_gross_mwh = float(dff_agg["gross_generation_mwh"].sum())
total_curt_mwh = float(dff_agg["curtailment_mwh"].sum())
curt_rate = (total_curt_mwh / total_gross_mwh) if total_gross_mwh > 0 else 0.0

total_revenue = float(dff_agg["revenue"].sum())
total_opex = float(dff_fin["opex"].sum())
total_profit = float(dff_fin["profit"].sum())
profit_margin = (total_profit / total_revenue) if total_revenue > 0 else 0.0

total_avoided = float(dff_agg["avoided_co2_tonnes"].sum())
avg_cf = float(dff_agg["capacity_factor"].mean()) if len(dff_agg) else 0.0
avg_avail = float(dff_agg["availability"].mean()) if len(dff_agg) else 0.0
outlier_pct = float(dff["is_any_outlier"].mean()) if len(dff) else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Net Generation (MWh)", format_number(total_net_mwh))
k2.metric("Curtailment (MWh)", format_number(total_curt_mwh))
k3.metric("Curtailment Rate", f"{curt_rate:.2%}")
k4.metric("Revenue", f"{total_revenue:,.0f}")
k5.metric("OPEX (Est.)", f"{total_opex:,.0f}")
k6.metric("Profit (Est.)", f"{total_profit:,.0f}")

k7, k8, k9 = st.columns(3)
k7.metric("Profit Margin (Est.)", f"{profit_margin:.2%}")
k8.metric("Avoided CO₂ (tonnes)", format_number(total_avoided))
k9.metric("Outlier %", f"{outlier_pct:.2%}")

insight_box(
    "Business insight (KPI summary)",
    [
        "Curtailment impacts both sustainability outcomes and financial outcomes (lost generation + lost revenue).",
        "Estimated OPEX and profit help prioritize interventions where the payback is highest.",
        "Use outlier % to judge how much extreme behavior is present in the filtered period.",
    ],
)

st.markdown("---")

# =============================
# FINANCE SECTION
# =============================
st.header("Finance")

# Income distribution
f1, f2 = st.columns(2)
with f1:
    fig = px.histogram(dff_fin, x="revenue", color="technology", nbins=40, title="Income Distribution (Revenue per plant-day)")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Shows how volatile income is across plant-days.", "A long tail indicates occasional high-price or high-output days."])

with f2:
    fig = px.box(dff_fin, x="technology", y="revenue", points="outliers", title="Revenue Distribution by Technology")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Helps identify which technologies have stable vs volatile revenues.", "Use for contract strategy and hedging decisions."])

# Expenditure and profitability
f3, f4 = st.columns(2)
with f3:
    fig = px.histogram(dff_fin, x="opex", color="technology", nbins=40, title="Expenditure Distribution (OPEX per plant-day)")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Higher OPEX days are often linked to downtime (maintenance) or inherently higher variable costs (biomass).", "Use for O&M budgeting and cost control."])

with f4:
    fig = px.histogram(dff_fin, x="profit", color="technology", nbins=40, title="Profit Distribution (Revenue − OPEX per plant-day)")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Negative/low-profit tails show when costs overwhelm revenue.", "Investigate which plants/tech drive profit compression."])

# Unit economics
f5, f6 = st.columns(2)
with f5:
    fig = px.box(dff_fin, x="technology", y="cost_per_mwh", points="outliers", title="Cost per MWh by Technology")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Cost per MWh is a clean operational KPI for benchmarking.", "Target cost reductions where cost/MWh is structurally higher."])

with f6:
    fig = px.box(dff_fin, x="technology", y="profit_per_mwh", points="outliers", title="Profit per MWh by Technology")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Profit per MWh highlights where value is created, independent of scale.", "Useful for portfolio optimization and expansion decisions."])

# Energy products (product mix)
st.subheader("Energy Products (Product Mix)")

prod_mix_mwh = product_df.groupby("energy_product", as_index=False)["product_mwh"].sum()
prod_mix_rev = product_df.groupby("energy_product", as_index=False)["product_revenue"].sum()

p1, p2 = st.columns(2)
with p1:
    fig = px.pie(prod_mix_mwh, names="energy_product", values="product_mwh", title="Product Mix by Volume (MWh)")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Shows how delivered energy is allocated across product types.", "Product diversification reduces exposure to single-market pricing."])

with p2:
    fig = px.pie(prod_mix_rev, names="energy_product", values="product_revenue", title="Product Mix by Revenue")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Revenue mix can differ from volume mix due to price premia.", "Supports pricing strategy and contract discussions."])

prod_month = product_df.copy()
prod_month["month"] = pd.to_datetime(prod_month["date"]).dt.to_period("M").astype(str)
prod_month = prod_month.groupby(["month", "energy_product"], as_index=False).agg(
    product_mwh=("product_mwh", "sum"),
    product_revenue=("product_revenue", "sum"),
)

fig = px.bar(prod_month.sort_values("month"), x="month", y="product_revenue", color="energy_product", title="Monthly Revenue by Energy Product (Stacked)")
st.plotly_chart(fig, use_container_width=True)
insight_box("Business insight", ["Highlights seasonality in product revenue.", "Use to plan contract volumes and target premium products during high-output months."])

st.markdown("---")

# =============================
# SUSTAINABILITY SECTION (kept, concise)
# =============================
st.header("Sustainability & Operations")

mix_gen = dff_agg.groupby("technology", as_index=False)["net_generation_mwh"].sum()
mix_co2 = dff_agg.groupby("technology", as_index=False)["avoided_co2_tonnes"].sum()
mix_curt = dff_agg.groupby("technology", as_index=False)["curtailment_mwh"].sum()

s1, s2, s3 = st.columns(3)
with s1:
    st.plotly_chart(px.pie(mix_gen, names="technology", values="net_generation_mwh", title="Generation Mix (Technology)"),
                    use_container_width=True)
    insight_box("Business insight", ["Where generation is concentrated is where performance improvements scale impact.", "Concentration also implies operational risk concentration."])
with s2:
    st.plotly_chart(px.pie(mix_co2, names="technology", values="avoided_co2_tonnes", title=f"Avoided CO₂ Mix ({baseline})"),
                    use_container_width=True)
    insight_box("Business insight", ["Supports ESG reporting and impact communication.", "Changing baseline alters avoided CO₂ magnitude."])
with s3:
    st.plotly_chart(px.pie(mix_curt, names="technology", values="curtailment_mwh", title="Curtailment Share (Donut)", hole=0.45),
                    use_container_width=True)
    insight_box("Business insight", ["Curtailment is a monetizable loss (lost revenue + lost impact).", "Prioritize mitigation where curtailment is concentrated."])

# Trends
daily = dff_agg.groupby("date", as_index=False).agg(
    net_generation_mwh=("net_generation_mwh", "sum"),
    gross_generation_mwh=("gross_generation_mwh", "sum"),
    curtailment_mwh=("curtailment_mwh", "sum"),
    avoided_co2_tonnes=("avoided_co2_tonnes", "sum"),
    revenue=("revenue", "sum"),
    opex=("opex", "sum"),
    profit=("profit", "sum"),
    grid_intensity_gco2_per_kwh=("grid_intensity_gco2_per_kwh", "mean"),
)
daily["curtailment_rate"] = safe_div(daily["curtailment_mwh"], daily["gross_generation_mwh"])
daily = daily.sort_values("date")
daily["net_7d_ma"] = daily["net_generation_mwh"].rolling(7, min_periods=1).mean()

t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(px.line(daily, x="date", y=["net_generation_mwh", "net_7d_ma"], title="Net Generation (Daily + 7D MA)"),
                    use_container_width=True)
    insight_box("Business insight", ["Sustained drops indicate outages, resource dips, or constraints.", "Use for operational monitoring."])
with t2:
    fig = px.line(daily, x="date", y="curtailment_rate", title="Curtailment Rate Trend")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Sustained curtailment increases indicate congestion risk.", "Align maintenance and storage dispatch to reduce opportunity cost."])

# Heatmaps
st.subheader("Heatmaps")

dff_agg["month"] = dff_agg["date"].dt.to_period("M").astype(str)
dff_agg["dow"] = pd.Categorical(dff_agg["date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)

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
    fig = px.imshow(hm_gen_pivot, aspect="auto", title="Heatmap: Net Generation (Month × Day-of-week)")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Find seasonality and weekly patterns.", "Schedule maintenance in lower-output periods."])
with h2:
    fig = px.imshow(hm_curt_pivot, aspect="auto", title="Heatmap: Curtailment Rate (Month × Day-of-week)")
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Pinpoints recurring constraint windows.", "Useful for storage and congestion management."])

st.markdown("---")

# =============================
# ML SECTION
# =============================
st.header("Machine Learning: Curtailment Event Prediction")

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
        insight_box("Business insight", ["Higher AUC means stronger early warning of curtailment events.", "Tune decision threshold to balance false alarms vs missed events."])

    with c2:
        cm = pd.DataFrame([[TN, FP], [FN, TP]], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        insight_box("Business insight", ["False negatives = missed high-curtailment days (risk).", "False positives = unnecessary interventions (cost)."])

    feat_names = X.columns.tolist()
    imp = np.abs(w)
    imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False).head(20)

    fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Feature Importance (|coef|, standardized)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Business insight", ["Top features indicate which signals to monitor to anticipate curtailment.", "Use these drivers to build operational playbooks and alerting."])
