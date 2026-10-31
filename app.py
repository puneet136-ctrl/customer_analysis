"""
Customer Behaviour Lab - ASSP mini project dashboard
Run with:  streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import stats_utils as su
from generate_sample_data import make_sample

st.set_page_config(page_title="Customer Behaviour Lab", page_icon="🛒", layout="wide")

STATE_COLORS = {"Active": "#2e9e6b", "Regular": "#f0a030", "Inactive": "#d9534f"}
TEMPLATE = "plotly_white"


# ----------------------------------------------------------------- helpers
def show(fig):
    fig.update_layout(template=TEMPLATE, margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def verdict(p, alpha):
    if p < alpha:
        return f"p = {p:.2g} < α = {alpha}  →  **Reject H₀**"
    return f"p = {p:.2g} ≥ α = {alpha}  →  **Fail to reject H₀**"


def effect_label(v, kind):
    v = abs(v)
    cuts = (0.2, 0.5, 0.8) if kind == "d" else (0.01, 0.06, 0.14)
    return ("negligible", "small", "medium", "large")[sum(v >= c for c in cuts)]


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner="Generating sample data…")
def load_sample():
    return make_sample()


@st.cache_data(show_spinner="Reading file…")
def read_upload(name, content):
    bio = io.BytesIO(content)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    try:
        return pd.read_csv(bio, encoding="utf-8")
    except UnicodeDecodeError:
        bio.seek(0)
        return pd.read_csv(bio, encoding="ISO-8859-1")


@st.cache_data(show_spinner="Cleaning transactions…")
def prepare(raw):
    return su.clean_transactions(raw)


@st.cache_data(show_spinner="Building customer states & Markov chain…")
def build_model(df, active_days, inactive_days):
    snapshot = df["date"].max() + pd.Timedelta(days=1)
    cust = su.customer_table(df, snapshot, active_days, inactive_days)
    S = su.monthly_states(df, active_days, inactive_days)
    C = su.transition_counts(S)
    P = su.transition_matrix(C)
    return snapshot, cust, S, C, P


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛒 Customer Behaviour Lab")
    src = st.radio("Data source", ["Built-in sample data", "Upload my dataset"])
    upload = None
    if src == "Upload my dataset":
        upload = st.file_uploader("CSV or Excel (Online Retail format)", type=["csv", "xlsx", "xls"])
        st.caption("Needs: invoice/order id, customer id, date, quantity, unit price "
                   "(country & product optional).")
    st.subheader("Customer-state rule")
    active_days = st.slider("Active if last purchase ≤ (days ago)", 7, 90, 30)
    inactive_days = st.slider("Inactive if last purchase > (days ago)",
                              active_days + 30, 365, max(90, active_days + 30))
    alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05)
    st.caption("Regular = in between the two thresholds.")

if src == "Upload my dataset":
    if upload is None:
        st.info("Upload a dataset in the sidebar, or switch to the built-in sample data.")
        st.stop()
    raw = read_upload(upload.name, upload.getvalue())
else:
    raw = load_sample()

try:
    df = prepare(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

if df["date"].dt.to_period("M").nunique() < 4:
    st.error("Need at least 4 months of data to estimate customer transitions.")
    st.stop()

snapshot, cust, S, C, P = build_model(df, active_days, inactive_days)

st.title("Customer Behaviour Lab")
st.caption("Measuring, modelling and testing e-commerce customer behaviour · "
           f"{len(df):,} cleaned transactions · {cust.shape[0]:,} customers · "
           f"{df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}")

tabs = st.tabs(["📊 Overview", "📈 Distributions", "🔄 States & Markov chain", "🧪 Hypothesis tests",
                "🎛️ What-if simulator", "🔎 Customer lookup", "✨ Extras"])

# =================================================================== OVERVIEW
with tabs[0]:
    spend = cust["total_spend"]
    c = st.columns(5)
    c[0].metric("Customers", f"{len(cust):,}")
    c[1].metric("Orders", f"{df['invoice'].nunique():,}")
    c[2].metric("Revenue", f"{spend.sum():,.0f}")
    c[3].metric("Avg order value", f"{spend.sum() / df['invoice'].nunique():,.1f}")
    c[4].metric("Inactive customers", f"{(cust['state'] == 'Inactive').mean():.0%}")

    pop, cum, gini = su.lorenz_gini(spend.values)
    top10 = spend.sort_values(ascending=False).head(max(1, int(0.1 * len(spend)))).sum() / spend.sum()
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=pop, y=cum, name="Lorenz curve", fill="tozeroy", line_color="#3b6fd4")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Perfect equality", line=dict(dash="dash", color="grey"))
        fig.update_layout(title=f"Spending concentration · Gini = {gini:.2f}",
                          xaxis_title="Cumulative share of customers",
                          yaxis_title="Cumulative share of revenue")
        show(fig)
        st.caption(f"The top 10% of customers generate **{top10:.0%}** of revenue.")
    with right:
        monthly = df.groupby(df["date"].dt.to_period("M").astype(str))["amount"].sum().reset_index()
        monthly.columns = ["Month", "Revenue"]
        show(px.bar(monthly, x="Month", y="Revenue", title="Monthly revenue"))

    left, right = st.columns(2)
    with left:
        counts = cust["state"].value_counts().reindex(su.STATES).reset_index()
        counts.columns = ["State", "Customers"]
        show(px.bar(counts, x="State", y="Customers", color="State", color_discrete_map=STATE_COLORS,
                    title="Customers by current state"))
    with right:
        ctry = df.groupby("country")["amount"].sum().sort_values(ascending=False).head(8).reset_index()
        ctry.columns = ["Country", "Revenue"]
        show(px.bar(ctry, x="Revenue", y="Country", orientation="h", title="Top countries by revenue"))

    st.download_button("⬇️ Download customer-level table (CSV)", cust.to_csv().encode(),
                       "customer_measures.csv", "text/csv")

# =================================================================== DISTRIBUTIONS
with tabs[1]:
    which = st.radio("Variable", ["Total spend → Lognormal", "Monthly order count → Poisson / NegBin",
                                  "Days between purchases → Exponential / Weibull"], horizontal=True)

    if which.startswith("Total"):
        x = cust["total_spend"].values
        fit = su.fit_lognormal(x)
        hi = np.percentile(x, 99)
        xs = np.linspace(x.min(), hi, 400)
        fig = go.Figure()
        fig.add_histogram(x=x[x <= hi], histnorm="probability density", nbinsx=60, name="Observed",
                          marker_color="#9db8ef")
        fig.add_scatter(x=xs, y=stats.lognorm.pdf(xs, fit["sigma"], 0, fit["scale"]), name="Lognormal fit",
                        line=dict(color="#d9534f", width=3))
        fig.update_layout(title="Total spend per customer (clipped at 99th percentile)",
                          xaxis_title="Total spend", yaxis_title="Density")
        l, r = st.columns([2, 1])
        with l:
            show(fig)
        with r:
            (osm, osr), _ = stats.probplot(np.log(x[x > 0]), dist="norm")
            qq = go.Figure()
            qq.add_scatter(x=osm, y=osr, mode="markers", name="log(spend)", marker=dict(size=4))
            qq.add_scatter(x=[osm.min(), osm.max()],
                           y=[fit["mu"] + fit["sigma"] * osm.min(), fit["mu"] + fit["sigma"] * osm.max()],
                           mode="lines", name="Perfect lognormal", line=dict(color="#d9534f"))
            qq.update_layout(title="QQ-plot of log(spend)", xaxis_title="Theoretical normal quantiles",
                             yaxis_title="Sample log(spend)")
            show(qq)
        m = st.columns(4)
        m[0].metric("μ (log scale)", f"{fit['mu']:.2f}")
        m[1].metric("σ (log scale)", f"{fit['sigma']:.2f}")
        m[2].metric("KS statistic", f"{fit['ks_stat']:.3f}")
        m[3].metric("KS p-value", f"{fit['ks_p']:.3g}")
        st.markdown("**Goodness of fit (KS test)** — H₀: spend is lognormal. " + verdict(fit["ks_p"], alpha))
        st.caption("KS p-values are optimistic when parameters are estimated from the same data; "
                   "treat them together with the QQ-plot and the AIC comparison below.")
        st.markdown("**Model selection — which distribution describes spend best?** (lower AIC = better)")
        table(su.compare_spend_models(x).round(1))

    elif which.startswith("Monthly"):
        counts = su.monthly_order_counts(df)
        r = su.poisson_vs_negbin(counts)
        fig = go.Figure()
        fig.add_bar(x=r["k"], y=r["empirical"], name="Observed", marker_color="#9db8ef")
        fig.add_scatter(x=r["k"], y=r["poisson_pmf"], name="Poisson", mode="lines+markers",
                        line=dict(color="#d9534f"))
        if not np.isnan(r["aic_nb"]):
            fig.add_scatter(x=r["k"], y=r["nb_pmf"], name="Negative binomial", mode="lines+markers",
                            line=dict(color="#2e9e6b"))
        fig.update_layout(title="Orders per customer per month (zero months included)",
                          xaxis_title="Orders in a month", yaxis_title="Probability")
        show(fig)
        m = st.columns(4)
        m[0].metric("Mean", f"{r['mean']:.2f}")
        m[1].metric("Variance", f"{r['var']:.2f}")
        m[2].metric("Dispersion (Var/Mean)", f"{r['dispersion']:.2f}")
        m[3].metric("AIC: Poisson vs NegBin",
                    f"{r['aic_poisson']:,.0f} vs {r['aic_nb']:,.0f}" if not np.isnan(r["aic_nb"]) else "n/a")
        if r["dispersion"] > 1.2:
            st.warning(f"**Overdispersion detected** (Var/Mean = {r['dispersion']:.2f}; a Poisson "
                       "model needs ≈ 1). Customers differ in their buying rate, so a pure Poisson "
                       "model is too narrow — the negative binomial (a Poisson with gamma-distributed "
                       "rate) fits better.")
        else:
            st.success("Variance ≈ mean, so a Poisson count model is reasonable.")
        if not np.isnan(r["chi2_p"]):
            st.markdown(f"**Chi-square GOF for Poisson** (χ² = {r['chi2']:.1f}, df = {r['chi2_dof']}) — "
                        "H₀: counts are Poisson. " + verdict(r["chi2_p"], alpha))

    else:
        gaps = su.interpurchase_gaps(df)
        e = su.fit_exponential_weibull(gaps)
        hi = np.percentile(gaps, 99)
        xs = np.linspace(0.5, hi, 300)
        fig = go.Figure()
        fig.add_histogram(x=gaps[gaps <= hi], histnorm="probability density", nbinsx=60, name="Observed",
                          marker_color="#9db8ef")
        fig.add_scatter(x=xs, y=stats.expon.pdf(xs, 0, e["mean_gap"]), name="Exponential",
                        line=dict(color="#d9534f", width=3))
        fig.add_scatter(x=xs, y=stats.weibull_min.pdf(xs, e["weibull_shape"], 0, e["weibull_scale"]),
                        name="Weibull", line=dict(color="#2e9e6b", width=3))
        fig.update_layout(title="Waiting time between a customer's purchases",
                          xaxis_title="Days", yaxis_title="Density")
        show(fig)
        m = st.columns(4)
        m[0].metric("Mean gap (days)", f"{e['mean_gap']:.1f}")
        m[1].metric("Exponential rate λ", f"{e['rate']:.3f}/day")
        m[2].metric("KS p-value (Exp.)", f"{e['ks_p']:.3g}")
        m[3].metric("Weibull shape k", f"{e['weibull_shape']:.2f}")
        shape = e["weibull_shape"]
        kind = ("**k < 1 → bursty buying**: customers who just bought are likely to buy again soon, "
                "then drift away." if shape < 0.95 else
                "**k > 1 → regular rhythm**: purchases follow a schedule." if shape > 1.05 else
                "**k ≈ 1 → memoryless**, consistent with the exponential model.")
        st.info(kind + f"  AIC — Exponential: {e['aic_exp']:,.0f} · Weibull: {e['aic_weibull']:,.0f}.")
        st.markdown(f"**Memorylessness check:** P(wait > {e['t']:.0f} d) = {e['p_uncond']:.2f}, but "
                    f"P(wait > {e['s'] + e['t']:.0f} d | already waited {e['s']:.0f} d) = {e['p_cond']:.2f}. "
                    "For a true exponential these two numbers would be equal.")

# =================================================================== STATES
with tabs[2]:
    st.markdown("Customers are classified at every month-end by recency "
                f"(Active ≤ {active_days} d · Regular ≤ {inactive_days} d · Inactive beyond that). "
                "The chain counts how customers move from one month to the next.")
    l, r = st.columns([1, 1])
    with l:
        fig = px.imshow(P, text_auto=".2f", color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="To state (next month)", y="From state (this month)", color="P"),
                        title="Transition probability matrix P")
        show(fig)
        with st.expander("Raw transition counts"):
            table(C.astype(int))
    with r:
        pi = su.steady_state(P)
        fig = px.bar(pd.DataFrame({"State": list(pi.index), "Share": pi.values}), x="State", y="Share",
                     color="State", color_discrete_map=STATE_COLORS, title="Long-run (steady-state) distribution π")
        fig.update_yaxes(tickformat=".0%")
        show(fig)
        mfp = su.mean_first_passage(P)
        m = st.columns(3)
        m[0].metric("Retention: Active→Active", f"{P.loc['Active', 'Active']:.0%}")
        m[1].metric("Months to churn from Active", f"{mfp.get('Active', np.nan):.1f}")
        m[2].metric("Months to churn from Regular", f"{mfp.get('Regular', np.nan):.1f}")

    mk = su.markov_order_test(S)
    if not np.isnan(mk["p"]):
        st.markdown(f"**Is the process really Markov?** χ² = {mk['chi2']:.1f}, df = {mk['dof']}. "
                    "H₀: next state depends only on the current state. " + verdict(mk["p"], alpha))
        st.caption("Rejection means customer history matters beyond the current month — a useful, honest "
                   "limitation to report (e.g. longer-inactive customers are even less likely to return).")

    st.subheader("Customer flow between two months")
    months = list(S.columns)
    ia = st.select_slider("From month", options=months[:-1], value=months[-2])
    ib = months[months.index(ia) + 1]
    pc = su.pair_counts(S, ia, ib)
    labels = [f"{s} · {ia}" for s in su.STATES] + [f"{s} · {ib}" for s in su.STATES]
    src_, tgt_, val_ = [], [], []
    for i, s in enumerate(su.STATES):
        for j, t in enumerate(su.STATES):
            if pc.loc[s, t] > 0:
                src_.append(i)
                tgt_.append(3 + j)
                val_.append(int(pc.loc[s, t]))
    cols = [STATE_COLORS[s] for s in su.STATES] * 2
    fig = go.Figure(go.Sankey(node=dict(label=labels, color=cols, pad=20),
                              link=dict(source=src_, target=tgt_, value=val_)))
    fig.update_layout(title=f"{ia} → {ib}")
    show(fig)

    over_time = S.apply(lambda col: col.value_counts()).reindex(su.STATES).fillna(0).T
    fig = px.area(over_time, color_discrete_map=STATE_COLORS, title="Customers in each state over time")
    fig.update_layout(xaxis_title="Month-end", yaxis_title="Customers")
    show(fig)

# =================================================================== TESTS
with tabs[3]:
    cA, cB = st.columns(2)
    group_by = cA.radio("Compare customers grouped by", ["Customer state", "Country (top 5)"], horizontal=True)
    measure = cB.radio("Measure", ["Total spend", "log(Total spend)"], horizontal=True)
    vals = cust["total_spend"] if measure == "Total spend" else np.log(cust["total_spend"])
    if group_by == "Customer state":
        key = cust["state"]
        order = su.STATES
    else:
        order = list(cust["country"].value_counts().head(5).index)
        key = cust["country"]
    groups = {g: vals[key == g].values for g in order if (key == g).sum() >= 3}
    if len(groups) < 2:
        st.warning("Not enough groups with ≥ 3 customers. Try the other grouping.")
    else:
        st.subheader("1 · Two-sample tests")
        names = list(groups)
        a_name = st.selectbox("Group A", names, index=0)
        b_name = st.selectbox("Group B", names, index=1)
        if a_name == b_name:
            st.info("Pick two different groups.")
        else:
            a, b = groups[a_name], groups[b_name]
            tt, ft = su.t_test(a, b), su.f_test(a, b)
            l, r = st.columns(2)
            with l:
                st.markdown(f"**Welch t-test** · H₀: μ_A = μ_B · H₁: μ_A ≠ μ_B")
                st.write(f"t = {tt['t']:.2f},  mean difference = {tt['diff']:.2f}  "
                         f"(95% CI {tt['ci'][0]:.2f} to {tt['ci'][1]:.2f})")
                st.markdown(verdict(tt["p"], alpha))
                st.caption(f"Cohen's d = {tt['d']:.2f} ({effect_label(tt['d'], 'd')} effect). "
                           f"Pooled-variance t-test gives p = {tt['p_pooled']:.2g}.")
            with r:
                st.markdown("**Two-sample F-test** · H₀: σ²_A = σ²_B · H₁: σ²_A ≠ σ²_B")
                st.write(f"F = {ft['F']:.2f}  (df = {ft['dfn']}, {ft['dfd']})")
                st.markdown(verdict(ft["p"], alpha))

        st.subheader("2 · One-way ANOVA across all groups")
        an = su.anova_suite(groups)
        m = st.columns(5)
        m[0].metric("F statistic", f"{an['F']:.2f}")
        m[1].metric("p-value", f"{an['p']:.2g}")
        m[2].metric("η² (effect size)", f"{an['eta2']:.3f}", effect_label(an["eta2"], "eta"), delta_color="off")
        m[3].metric("Levene p (equal var.)", f"{an['levene_p']:.2g}")
        m[4].metric("Kruskal-Wallis p", f"{an['kruskal_p']:.2g}")
        st.markdown(f"H₀: all group means equal · H₁: at least one differs · "
                    f"F({an['df_between']}, {an['df_within']}) — " + verdict(an["p"], alpha))
        with st.expander("Assumption checks (normality per group — Shapiro-Wilk)"):
            sh = pd.DataFrame({"Group": list(an["shapiro"]), "Shapiro p": list(an["shapiro"].values())})
            sh["Normal at α?"] = np.where(sh["Shapiro p"] >= alpha, "Yes", "No")
            table(sh.round(4))
            st.caption("If groups are clearly non-normal (typical for raw spend), prefer log(spend) "
                       "or lean on the non-parametric Kruskal-Wallis result.")
        if an["tukey"] is not None:
            st.markdown("**Post-hoc: Tukey HSD** — which pairs differ?")
            tk = an["tukey"].copy()
            tk["Significant"] = np.where(tk["p (Tukey)"] < alpha, "✅", "—")
            table(tk.round(4))

        st.subheader("3 · Bootstrap 95% confidence intervals for the mean")
        rows = []
        for g, v in groups.items():
            mean, lo, hi = su.bootstrap_mean_ci(v)
            rows.append((g, mean, lo, hi))
        bt = pd.DataFrame(rows, columns=["Group", "Mean", "Lo", "Hi"])
        fig = go.Figure(go.Scatter(x=bt["Group"], y=bt["Mean"], mode="markers",
                                   marker=dict(size=11, color="#3b6fd4"),
                                   error_y=dict(type="data", symmetric=False,
                                                array=bt["Hi"] - bt["Mean"], arrayminus=bt["Mean"] - bt["Lo"])))
        fig.update_layout(title=f"Mean of {measure.lower()} with bootstrap CI", yaxis_title=measure)
        show(fig)
        st.caption("Non-overlapping intervals agree with a significant ANOVA; overlapping intervals "
                   "suggest the groups are hard to tell apart.")

# =================================================================== WHAT-IF
with tabs[4]:
    st.markdown("Use the fitted Markov chain as a simulator: what happens to the customer base and "
                "revenue if marketing improves win-back or reduces churn?")
    c1, c2, c3 = st.columns(3)
    boost = c1.slider("Win-back: +pp in P(Inactive → Regular)", 0.0, 0.20, 0.05, 0.01)
    cut = c2.slider("Churn reduction (relative cut in moves to Inactive)", 0.0, 0.5, 0.20, 0.05)
    horizon = c3.slider("Horizon (months)", 3, 36, 12)

    P2 = su.adjust_matrix(P, boost, cut)
    start = S.iloc[:, -1].value_counts().reindex(su.STATES).fillna(0)
    base, scen = su.project(P, start, horizon), su.project(P2, start, horizon)
    st_spend = su.state_monthly_spend(df, S)
    rev_base, rev_scen = (base * st_spend).sum(axis=1), (scen * st_spend).sum(axis=1)

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_scatter(y=base["Active"], name="Baseline", line=dict(color="grey", dash="dash"))
        fig.add_scatter(y=scen["Active"], name="Scenario", line=dict(color="#2e9e6b", width=3))
        fig.update_layout(title="Expected Active customers", xaxis_title="Months ahead", yaxis_title="Customers")
        show(fig)
    with r:
        fig = go.Figure()
        fig.add_scatter(y=rev_base, name="Baseline", line=dict(color="grey", dash="dash"))
        fig.add_scatter(y=rev_scen, name="Scenario", line=dict(color="#3b6fd4", width=3))
        fig.update_layout(title="Expected monthly revenue", xaxis_title="Months ahead", yaxis_title="Revenue")
        show(fig)
    gain = rev_scen.sum() - rev_base.sum()
    pi0, pi1 = su.steady_state(P), su.steady_state(P2)
    m = st.columns(3)
    m[0].metric(f"Extra revenue over {horizon} months", f"{gain:,.0f}")
    m[1].metric("Long-run Active share", f"{pi1['Active']:.1%}", f"{(pi1['Active'] - pi0['Active']) * 100:+.1f} pp")
    m[2].metric("Active customers at horizon", f"{scen['Active'].iloc[-1]:,.0f}",
                f"{scen['Active'].iloc[-1] - base['Active'].iloc[-1]:+,.0f}")
    with st.expander("Scenario transition matrix"):
        table(P2.round(3))
    st.caption("Revenue per state is the historical average monthly spend of customers in that state. "
               "Projection = starting state counts × Pⁿ.")

# =================================================================== CUSTOMER LOOKUP
with tabs[5]:
    ids = list(cust.sort_values("total_spend", ascending=False).index)
    cid = st.selectbox("Customer ID (sorted by spend; type to search)", ids)
    row = cust.loc[cid]
    m = st.columns(5)
    m[0].metric("State now", row["state"])
    m[1].metric("Total spend", f"{row['total_spend']:,.0f}")
    m[2].metric("Orders", int(row["n_orders"]))
    m[3].metric("Days since last order", int(row["recency"]))
    m[4].metric("Country", row["country"])

    hist = S.loc[cid].dropna()
    tl = pd.DataFrame({"Month": hist.index, "State": hist.values})
    fig = px.scatter(tl, x="Month", y="State", color="State", color_discrete_map=STATE_COLORS,
                     category_orders={"State": su.STATES}, title="State timeline")
    fig.update_traces(marker=dict(size=14))
    show(fig)

    onehot = pd.Series([1.0 if s == row["state"] else 0.0 for s in su.STATES], index=su.STATES)
    fut = su.project(P, onehot, 3).iloc[1:]
    fut.index = ["+1 month", "+2 months", "+3 months"]
    st.markdown("**Predicted state probabilities** (from the Markov chain)")
    table(fut.round(3))
    st.metric("Probability of being Inactive within 3 months",
              f"{fut['Inactive'].iloc[-1]:.0%}")
    orders = (df[df["customer"] == cid].groupby("invoice")
              .agg(date=("date", "min"), items=("qty", "sum"), amount=("amount", "sum")).sort_values("date"))
    with st.expander("Order history"):
        table(orders.assign(amount=orders["amount"].round(2)))

# =================================================================== EXTRAS
with tabs[6]:
    st.subheader("When do customers buy? (orders by weekday × hour)")
    inv = df.drop_duplicates("invoice")
    heat = (inv.assign(day=inv["date"].dt.day_name(), hour=inv["date"].dt.hour)
               .groupby(["day", "hour"]).size().unstack(fill_value=0))
    days = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if d in heat.index]
    show(px.imshow(heat.loc[days], color_continuous_scale="YlOrRd", aspect="auto",
                   labels=dict(x="Hour of day", y="", color="Orders")))

    st.subheader("How long until a customer churns? (Kaplan-Meier survival)")
    churned = (cust["recency"] > inactive_days).astype(int)
    dur = np.where(churned == 1, (cust["last"] - cust["first"]).dt.days, cust["tenure_days"])
    km = su.kaplan_meier(dur, churned.values)
    fig = px.line(km, x="t", y="survival", line_shape="hv",
                  labels={"t": "Days since first purchase", "survival": "P(still active customer)"})
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    show(fig)
    st.caption(f"A customer is 'churned' once their last purchase is more than {inactive_days} days old; "
               "everyone else is right-censored at the end of the data.")

    st.subheader("Outlier customers (above the fitted lognormal's 99th percentile)")
    fit = su.fit_lognormal(cust["total_spend"].values)
    thr = stats.lognorm.ppf(0.99, fit["sigma"], 0, fit["scale"])
    flagged = cust[cust["total_spend"] > thr].sort_values("total_spend", ascending=False)
    st.write(f"Threshold = **{thr:,.0f}**; **{len(flagged)}** customers exceed it and account for "
             f"**{flagged['total_spend'].sum() / cust['total_spend'].sum():.0%}** of revenue "
             f"(a lognormal would predict only 1% of customers above it).")
    table(flagged[["total_spend", "n_orders", "recency", "state", "country"]].head(20).round(1))

with st.expander("📐 Method summary (for your report)"):
    st.latex(r"\text{Spend}_i=\sum_j Q_{ij}\,p_{ij},\qquad \text{Recency}_i=t_{\text{snapshot}}-t^{\text{last}}_i")
    st.latex(r"\text{Lognormal: } \ln X\sim\mathcal N(\mu,\sigma^2)\quad "
             r"\text{Poisson: } P(N=k)=\frac{e^{-\lambda}\lambda^k}{k!}\quad "
             r"\text{Exponential: } f(t)=\lambda e^{-\lambda t}")
    st.latex(r"P_{ij}=\frac{n_{ij}}{\sum_k n_{ik}},\qquad \pi P=\pi,\qquad (I-Q)\,m=\mathbf 1")
    st.latex(r"F=\frac{MS_{between}}{MS_{within}},\qquad \eta^2=\frac{SS_{between}}{SS_{total}}")
