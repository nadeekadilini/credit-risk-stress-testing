"""Credit Risk | Portfolio Analytics Dashboard
Run with: python -m streamlit run app.py
Uses exports from 01_credit_risk_EDA.ipynb in outputs/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Credit Risk | Portfolio Intelligence", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
NAVY, TEAL, BLUE, AMBER, RED = "#14283f", "#157e83", "#4c82b8", "#dfa33a", "#b85d5d"
sns.set_theme(style="whitegrid", font_scale=0.92)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
st.markdown("""<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1450px;}
h1,h2,h3 {color:#14283f; letter-spacing:-.02em;}
[data-testid="stMetric"] {background:#f3f7fa;border:1px solid #e2e9ef;border-radius:14px;padding:14px 16px;}
[data-testid="stMetricLabel"] {color:#4d6072;}
[data-testid="stSidebar"] {background:#f6f9fb;}
.small-note {font-size:0.86rem;color:#627487;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_output(name):
    path = OUT / name
    return pd.read_csv(path) if path.exists() else None

scenario = read_output("scenario_results_verified.csv")
metrics = read_output("model_metrics_verified.csv")
comparison = read_output("model_comparison_verified.csv")
cv = read_output("logistic_regression_cv_results.csv")
sensitivity = read_output("pd_lgd_sensitivity.csv")
holdout = read_output("holdout_predictions.csv")  # DIFFERENT starter pipeline

st.sidebar.markdown("### CREDIT RISK INTELLIGENCE")
page = st.sidebar.radio("Navigate", ["Overview", "Stress scenarios", "Sensitivity lab", "Model diagnostics", "Data & methodology"])
st.sidebar.divider()
st.sidebar.caption("UCI German Credit • 1,000 observations • 750 train / 250 test")
st.sidebar.caption("Verified notebook outputs used for ECL and model metrics. Original starter-pipeline probability file is separate.")

st.title("Credit Risk | Portfolio Intelligence")
st.caption("Statistical modelling · Scenario analysis · Portfolio loss sensitivity")
st.info("Research demonstration, not a live bank risk system. Bad-credit probability is only a PD proxy; LGD and stress effects are assumed. No estimated macroeconomic linkage or IFRS 9 compliance.")

if scenario is None or sensitivity is None:
    st.error("Verified scenario or sensitivity files not found. In your notebook, export `scenario_results_verified.csv` and `pd_lgd_sensitivity.csv` into the `outputs` folder.")
    st.stop()

scenario = scenario.copy()
scenario["Scenario"] = scenario["Scenario"].astype(str)
order = [x for x in ["Baseline", "Adverse", "Severe"] if x in set(scenario["Scenario"])]
scenario = scenario.set_index("Scenario").loc[order].reset_index()
if not order or "Baseline" not in order:
    st.error("Verified scenario table must include a Baseline row.")
    st.stop()

# Assumptions used in the user's verified notebook, not inferred from starter model.
assumed_multipliers = {"Baseline":1., "Adverse":1.5, "Severe":2.}
scenario["PD_Odds_Multiplier"] = scenario["Scenario"].map(assumed_multipliers)
base_ecl = float(scenario.loc[scenario.Scenario.eq("Baseline"), "Total_ECL"].iloc[0])
scenario["Increase_pct"] = (scenario.Total_ECL / base_ecl - 1) * 100
scenario["Additional_ECL"] = scenario.Total_ECL - base_ecl


def currency(v): return f"{v:,.0f}"
def pct(v): return f"{v:.1%}"
def tidy_chart(fig):
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def scenario_bars(df):
    fig, ax = plt.subplots(figsize=(9,4.2))
    colors = [BLUE, AMBER, RED][:len(df)]
    bars = ax.bar(df.Scenario, df.Total_ECL, color=colors, width=.58)
    ax.set_ylabel("Expected credit loss (dataset credit units)")
    ax.set_ylim(0, df.Total_ECL.max()*1.22)
    ax.grid(axis="y", alpha=.15)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.025, currency(bar.get_height()), ha="center", fontsize=10, weight="bold", color=NAVY)
    return fig


def pd_lines(df):
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(df))
    ax.plot(x, df.Average_PD*100, "o-", lw=2.6, color=BLUE, label="Average PD proxy")
    ax.plot(x, df.Exposure_Weighted_PD*100, "s-", lw=2.6, color=TEAL, label="Exposure-weighted PD proxy")
    ax.set_xticks(x, df.Scenario)
    ax.set_ylabel("Bad-credit probability proxy (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(df.Exposure_Weighted_PD.max()*120, 45))
    ax.grid(alpha=.16)
    return fig


def lgd_line(df):
    fig, ax = plt.subplots(figsize=(7,3.8))
    ax.plot(df.Scenario, df.LGD*100, "o-", lw=2.8, color=AMBER)
    ax.set_ylabel("Assumed LGD (%)")
    ax.set_ylim(0, max(70, df.LGD.max()*115))
    ax.grid(alpha=.16)
    return fig


def risk_heatmap(mat):
    fig, ax = plt.subplots(figsize=(10,5.8))
    sns.heatmap(mat, annot=True, fmt=",.0f", cmap="YlOrRd", linewidths=.6, cbar_kws={"label":"Expected credit loss"}, ax=ax)
    ax.set_xlabel("Assumed loss given default (LGD)")
    ax.set_ylabel("PD odds multiplier")
    return fig


def metric_row(*items):
    columns = st.columns(len(items))
    for col, (label, value, extra) in zip(columns, items):
        col.metric(label, value, help=extra)


if page == "Overview":
    selected = st.selectbox("Focus scenario", order, index=0)
    r = scenario.loc[scenario.Scenario.eq(selected)].iloc[0]
    metric_row(
        ("Illustrative portfolio", "750 loans", "In-sample training portfolio"),
        ("Portfolio exposure", "2,409,866", "Fixed total exposure from verified notebook, original credit units"),
        ("Selected ECL", currency(r.Total_ECL), "Illustrative PD-proxy × assumed LGD × exposure"),
        ("Change vs baseline", f"{r.Increase_pct:+.2f}%", "Scenario sensitivity, not calibrated macro forecast"),
    )
    a,b=st.columns([1.25,1])
    with a:
        st.subheader("Expected losses by scenario")
        tidy_chart(scenario_bars(scenario))
    with b:
        st.subheader("Portfolio risk by scenario")
        tidy_chart(pd_lines(scenario))
    st.subheader("Scenario scorecard")
    view=scenario[["Scenario","Average_PD","Exposure_Weighted_PD","LGD","Total_ECL","ECL_Rate","Increase_pct"]].copy()
    st.dataframe(view.style.format({"Average_PD":"{:.2%}","Exposure_Weighted_PD":"{:.2%}","LGD":"{:.0%}","Total_ECL":"{:,.0f}","ECL_Rate":"{:.2%}","Increase_pct":"{:+.2f}%"}), use_container_width=True, hide_index=True)

elif page == "Stress scenarios":
    st.subheader("Scenario comparison")
    st.caption("Fixed exposures. Assumed PD odds multipliers: 1.0× / 1.5× / 2.0×; assumed LGD: 45% / 50% / 60%.")
    a,b=st.columns([1.3,1])
    with a: tidy_chart(scenario_bars(scenario))
    with b: tidy_chart(lgd_line(scenario))
    a,b=st.columns(2)
    with a:
        st.subheader("Default risk proxy")
        tidy_chart(pd_lines(scenario))
    with b:
        st.subheader("Incremental loss vs baseline")
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(scenario.Scenario,scenario.Additional_ECL,color=[BLUE,AMBER,RED][:len(scenario)])
        ax.set_ylabel("Additional ECL (credit units)")
        ax.grid(axis="y",alpha=.15)
        tidy_chart(fig)
    st.dataframe(scenario.style.format({"Average_PD":"{:.2%}","Exposure_Weighted_PD":"{:.2%}","LGD":"{:.0%}","Total_ECL":"{:,.0f}","ECL_Rate":"{:.2%}","Increase_pct":"{:+.2f}%","Additional_ECL":"{:,.0f}","PD_Odds_Multiplier":"{:.2f}x"}),hide_index=True,use_container_width=True)

elif page == "Sensitivity lab":
    st.subheader("PD–LGD sensitivity heatmap")
    st.caption("This is the heatmap from your notebook, regenerated directly from your saved 30-scenario sensitivity table.")
    mat=sensitivity.pivot(index="PD_Odds_Multiplier", columns="LGD", values="Total_ECL").sort_index().sort_index(axis=1)
    tidy_chart(risk_heatmap(mat))
    st.divider()
    st.subheader("Interactive assumption explorer")
    a,b=st.columns(2)
    with a:
        mult=st.select_slider("PD odds multiplier", options=sorted(mat.index.tolist()), value=float(mat.index[0]))
    with b:
        lgd=st.select_slider("Assumed LGD",options=sorted(mat.columns.tolist()),value=min(mat.columns,key=lambda x:abs(x-.45)))
    selected_ecl=float(mat.loc[mult,lgd])
    metric_row(("Selected ECL",currency(selected_ecl),"Computed from verified sensitivity grid"),("Change vs baseline",f"{(selected_ecl/base_ecl-1)*100:+.2f}%","Baseline = 1.0× odds and 45% LGD"))
    c,d=st.columns(2)
    with c:
        st.subheader("ECL across LGD assumptions")
        fig,ax=plt.subplots(figsize=(7,4))
        for m in mat.index:
            ax.plot(mat.columns*100,mat.loc[m],marker="o",label=f"{m:g}× odds")
        ax.set_xlabel("Assumed LGD (%)");ax.set_ylabel("Illustrative ECL");ax.legend(fontsize=8,ncol=2,frameon=False);ax.grid(alpha=.15)
        tidy_chart(fig)
    with d:
        st.subheader("ECL across odds assumptions")
        fig,ax=plt.subplots(figsize=(7,4))
        for g in mat.columns:
            ax.plot(mat.index,mat[g],marker="o",label=f"LGD {g:.0%}")
        ax.set_xlabel("PD odds multiplier");ax.set_ylabel("Illustrative ECL");ax.legend(fontsize=8,ncol=2,frameon=False);ax.grid(alpha=.15)
        tidy_chart(fig)
    st.download_button("Download sensitivity CSV",sensitivity.to_csv(index=False).encode(),"pd_lgd_sensitivity.csv","text/csv")

elif page == "Model diagnostics":
    st.subheader("Holdout and cross-validation")
    if metrics is not None and {"Metric","Value"}.issubset(metrics.columns):
        mm=dict(zip(metrics.Metric,metrics.Value))
        metric_row(*[(name, f"{float(mm[name]):.4f}", "Verified notebook holdout metric") for name in ["ROC-AUC","Average Precision","Brier Score","Recall"] if name in mm])
        st.dataframe(metrics.style.format({"Value":"{:.4f}"}),hide_index=True,use_container_width=True)
    else: st.info("Verified metrics unavailable: export `model_metrics_verified.csv` from notebook.")
    if comparison is not None:
        st.subheader("Model benchmark")
        a,b=st.columns([1,1])
        with a:
            fig,ax=plt.subplots(figsize=(7,4))
            data=comparison.set_index("Model")[["ROC_AUC_Mean","Average_Precision"]]
            data.plot.bar(ax=ax,color=[BLUE,TEAL],rot=0)
            ax.set_ylim(0,1);ax.legend(["Mean ROC-AUC","Avg precision"],frameon=False)
            tidy_chart(fig)
        with b: st.dataframe(comparison.round(4),hide_index=True,use_container_width=True)
    if cv is not None:
        st.subheader("5-fold training cross-validation")
        st.dataframe(cv.round(4),hide_index=True,use_container_width=True)
    st.divider()
    st.subheader("Starter-model diagnostic plots (separate pipeline)")
    st.warning("The images and histogram below, when present, belong to the earlier starter pipeline, not necessarily to the verified notebook model. Do not present them as identical evaluations.")
    a,b=st.columns(2)
    with a:
        if (OUT/"roc.png").exists():st.image(str(OUT/"roc.png"),caption="Starter-model ROC")
    with b:
        if (OUT/"calibration.png").exists():st.image(str(OUT/"calibration.png"),caption="Starter-model calibration")
    if holdout is not None and "predicted_pd" in holdout.columns:
        st.subheader("Starter-model holdout risk distribution")
        p=pd.to_numeric(holdout.predicted_pd,errors="coerce").dropna()
        fig,ax=plt.subplots(figsize=(9,3.8))
        ax.hist(p,bins=np.linspace(0,1,11),color=BLUE,edgecolor="white")
        ax.set_xlabel("Predicted bad-credit probability");ax.set_ylabel("Observations")
        tidy_chart(fig)

else:
    st.subheader("Methods, model provenance and limitations")
    st.markdown("""
**Data:** Historical UCI German Credit; 1,000 records, 20 predictors, target good/bad risk. Stratified 750/250 split.

**Baseline model:** L2-regularised logistic regression with standardised numeric and one-hot-encoded categorical predictors. Random Forest benchmark and 5-fold stratified CV.

**ECL illustration:** Loan credit amount is used as exposure proxy; the bad-credit probability acts as a PD proxy; LGD is assumed. Portfolio figures come from in-sample training predictions.

**Stress exercise:** Odds multipliers (1×, 1.5×, 2×) plus LGD (45%, 50%, 60%). These are *hypothetical parameter shocks*, not empirically estimated GDP, unemployment or interest-rate sensitivities.

**Limitations:** No default event timing, multi-period ECL, stage assignment, recoveries, observed LGD, discounting, model governance validation or calibrated macro-credit relationship. No production lending or regulatory use. Model fairness and use of sensitive predictors would require further review.

**Provenance:** Verified notebook output CSVs are authoritative for displayed ECL, model scores and sensitivity. The earlier `src/model.py` pipeline created separate plots and holdout predictions, labelled accordingly in Model Diagnostics.
""")
    st.subheader("Available files")
    for name in ["scenario_results_verified.csv","model_metrics_verified.csv","model_comparison_verified.csv","logistic_regression_cv_results.csv","pd_lgd_sensitivity.csv","holdout_predictions.csv"]:
        st.write(f"{'✅' if (OUT/name).exists() else '—'} `{name}`")

st.divider()
st.caption("Independent portfolio demonstration • All monetary amounts are dataset credit units, not AUD • Source: UCI German Credit")
