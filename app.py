import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from pathlib import Path

st.set_page_config(
    page_title="Philippine ICT Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

px.defaults.template = "plotly_dark"



# ---------- Theme ----------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background-color: #0B0F19; color: #F1F5F9; }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px; }
[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #263449; }
h1, h2, h3, h4, h5, p, label, [data-testid="stMarkdownContainer"] { color: #F1F5F9; }
.hero { background: linear-gradient(120deg,#14243b 0%,#244b72 100%); color:white; padding:26px 30px; border-radius:16px; margin-bottom:22px; }
.hero h1 { color:white; font-size:2rem; margin:0 0 7px 0; }
.hero p { color:#DBE7F5; font-size:1rem; margin:0; }
.section-kicker { color:#60A5FA; font-size:.78rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; }
.metric-card { background:#161F2E; border:1px solid #2B3A50; padding:18px; border-radius:13px; min-height:125px; height:100%; box-sizing:border-box; }
.metric-label { color:#AAB8CA; font-size:.82rem; margin-bottom:10px; }
.metric-value { color:#FFFFFF; font-size:1.55rem; font-weight:750; overflow-wrap:anywhere; }
.metric-note { color:#9AAAC0; font-size:.78rem; margin-top:8px; }
.info-panel { background-color:#161F2E !important; color:#E2E8F0 !important; border:1px solid #2B3A50; border-radius:12px; padding:18px 20px; line-height:1.7; }
.info-panel b { color:#FFFFFF !important; }
.workspace-card { background:#161F2E; border:1px solid #2B3A50; border-radius:13px; padding:20px; min-height:155px; height:100%; box-sizing:border-box; }
.workspace-card h4 { color:#FFFFFF; margin-top:0; }
.workspace-card p { color:#B8C5D6; line-height:1.6; }
.small-note { color:#AAB8CA !important; font-size:.86rem; }
div.stButton > button { border-radius:10px; min-height:44px; font-weight:600; }
[data-testid="stDataFrame"], [data-testid="stTable"] { background-color:#161F2E; }
[data-testid="stExpander"] { background-color:#161F2E; border:1px solid #2B3A50; border-radius:10px; }
[data-testid="stAlert"] { background-color:#182538; color:#F1F5F9; border:1px solid #34465E; border-radius:10px; }
input, textarea { background-color:#161F2E !important; color:#FFFFFF !important; }
[data-baseweb="select"] > div { background-color:#161F2E !important; color:#FFFFFF !important; }
[data-testid="stPlotlyChart"] { background-color:#0B0F19; border-radius:10px; }
hr { border-color:#2B3A50; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
RAW_DEFAULT = ROOT / "core_ict_indicators_raw_for_descriptive_analytics.csv"
ML_DEFAULT = ROOT / "core_ict_indicators_cleaned_for_ml.csv"

@st.cache_data
def load_csv(uploaded, fallback):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if fallback.exists():
        return pd.read_csv(fallback)
    return None

def clean_label(value):
    return str(value).replace("..", "").strip()

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def section_header(kicker, title, subtitle):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="small-note">{subtitle}</p>', unsafe_allow_html=True)

# ---------- Sidebar / data ----------
with st.sidebar:
    st.markdown("## 📊 ICT Analytics")
    st.caption("Philippine establishments • ICT indicators")
    st.divider()
    page_options = ["Main Dashboard", "Descriptive Analysis", "Predictive Analysis", "Supervised Model"]
    if "page" not in st.session_state:
        st.session_state.page = "Main Dashboard"
    page = st.radio("WORKSPACE", page_options, key="page")
    st.divider()
    st.markdown("**DATA SOURCES**")
    raw_upload = st.file_uploader("Raw descriptive dataset (CSV)", type=["csv"], key="rawfile")
    ml_upload = st.file_uploader("Structured ML dataset (CSV)", type=["csv"], key="mlfile")
    st.caption("Default files are loaded automatically when they are in the same folder as this app.")
    st.divider()
    st.caption("Analysis period in source data: 2010–2021 (selected survey years).")

raw = load_csv(raw_upload, RAW_DEFAULT)
ml = load_csv(ml_upload, ML_DEFAULT)

if raw is None or ml is None:
    st.warning("Upload both CSV files in the sidebar, or place the supplied CSV files beside app.py.")
    st.stop()

raw.columns = raw.columns.astype(str).str.strip()
ml.columns = ml.columns.astype(str).str.strip()
if "Section Description" not in raw.columns:
    st.error("The raw dataset must contain a 'Section Description' column.")
    st.stop()
raw["Sector"] = raw["Section Description"].map(clean_label)
sector = raw[raw["Sector"].str.lower() != "philippines"].copy()
years = [2010, 2013, 2015, 2017, 2019, 2021]
metric_options = ["Establishments with internet access", "Establishments with computers and communication equipment", "Establishments with web presence", "Establishments with e-commerce via the internet", "Establishments with intranet", "Establishments with wired local area network", "Establishments with wireless local area network"]
metric_prefix = {
    "Establishments with internet access":"Establishments with internet access",
    "Establishments with computers and communication equipment":"Establishments with computers and communication equipment",
    "Establishments with web presence":"Establishments with web presence",
    "Establishments with e-commerce via the internet":"Establishments with e-commerce via the internet",
    "Establishments with intranet":"Establishments with intranet",
    "Establishments with wired local area network":"Establishments with wired local area network",
    "Establishments with wireless local area network":"Establishments with wireless local area network"
}
def year_col(prefix, year):
    col = f"{prefix} {year}"
    return col if col in raw.columns else None

def value_col(df, prefix, year):
    col = f"{prefix} {year}"
    return col if col in df.columns else None

# ---------- Header and navigation ----------
st.markdown('<div class="hero"><h1>Philippine ICT Indicators Analytics</h1><p>Explore digital readiness, internet adoption, forecasting trends, and supervised classification results across industry sectors.</p></div>', unsafe_allow_html=True)
st.markdown("### Explore the workspace")
nav_cols = st.columns(4)
def navigate_page(label):
    st.session_state.page = label

for col, label, icon in zip(nav_cols, page_options, ["⌂", "◫", "↗", "◎"]):
    with col:
        st.button(
            f"{icon}  {label}",
            use_container_width=True,
            type="primary" if page == label else "secondary",
            on_click=navigate_page,
            args=(label,)
        )

# ---------- Main Dashboard ----------
if page == "Main Dashboard":
    section_header("OVERVIEW", "Main Dashboard", "A concise overview of ICT adoption indicators reported by Philippine establishments.")
    latest = 2021
    internet = value_col(raw, "Establishments with internet access", latest)
    computers = value_col(raw, "Establishments with computers and communication equipment", latest)
    web = value_col(raw, "Establishments with web presence", latest)
    establishments = value_col(raw, "Number of establishments", latest)
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Sectors represented", f"{len(sector):,}", "Excludes Philippines aggregate row")
    with c2: metric_card("Internet access • 2021", f"{sector[internet].mean():.1f}%" if internet else "—", "Average across sector rows")
    with c3: metric_card("Computer equipment • 2021", f"{sector[computers].mean():.1f}%" if computers else "—", "Average across sector rows")
    with c4: metric_card("Web presence • 2021", f"{sector[web].mean():.1f}%" if web else "—", "Average across sector rows")
    st.markdown("### ICT adoption snapshot")
    left,right = st.columns([1.15,1])
    with left:
        if internet:
            trend = pd.DataFrame({"Year":years, "Average internet access (%)":[sector[value_col(raw,"Establishments with internet access",y)].mean() if value_col(raw,"Establishments with internet access",y) else np.nan for y in years]})
            fig=px.line(trend,x="Year",y="Average internet access (%)",markers=True,title="Average internet access across sectors")
            fig.update_layout(height=340,margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig,use_container_width=True)
    with right:
        if web and computers:
            snap=pd.DataFrame({"Indicator":["Computer equipment","Internet access","Web presence"],"Average (%)":[sector[computers].mean(),sector[internet].mean(),sector[web].mean()]})
            fig=px.bar(snap,x="Average (%)",y="Indicator",orientation="h",text_auto=".1f",title="2021 digital readiness")
            fig.update_layout(height=340,margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig,use_container_width=True)
    st.markdown("### What each workspace contains")
    a, b, c = st.columns(3)
    with a:
        st.markdown("""
        <div class="workspace-card">
            <h4>📊 Descriptive Analysis</h4>
            <p>Summarizes sector distributions and relationships using charts and descriptive tables.</p>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="workspace-card">
            <h4>📈 Predictive Analysis</h4>
            <p>Extends the historical sector-average internet-access trend through 2026 using linear regression.</p>
        </div>
        """, unsafe_allow_html=True)
    with c:
        st.markdown("""
        <div class="workspace-card">
            <h4>🎯 Supervised Model</h4>
            <p>Classifies 2021 internet access as high or low and compares four supervised classifiers.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="info-panel"><b>Reading the data</b><br>Percentages are sector-level reported shares. The Philippines row is an overall aggregate and is excluded from sector averages and sector comparisons to avoid mixing the national total with individual industries.</div>',unsafe_allow_html=True)

# ---------- Descriptive ----------
elif page == "Descriptive Analysis":
    section_header("EXPLORE THE DATA", "Descriptive Analysis", "Explore the distribution, sector differences, and relationships in the original raw dataset.")
    metric = st.selectbox("Choose an ICT indicator", metric_options, index=0)
    prefix = metric_prefix[metric]
    available_years = [y for y in years if value_col(raw,prefix,y)]
    if not available_years:
        st.warning("This indicator is not available in the uploaded raw dataset.")
        st.stop()
    selected_year = st.select_slider("Reference year", options=available_years, value=available_years[-1])
    col = value_col(raw,prefix,selected_year)
    desc = sector[["Sector",col]].dropna().sort_values(col,ascending=False)
    total_col=value_col(raw,"Number of establishments",selected_year)
    c1,c2,c3=st.columns(3)
    with c1: metric_card("Sector average",f"{desc[col].mean():.1f}%","Unweighted mean of sector percentages")
    with c2: metric_card("Highest sector",clean_label(desc.iloc[0]["Sector"]),f"{desc.iloc[0][col]:.1f}%")
    with c3: metric_card("Lowest sector",clean_label(desc.iloc[-1]["Sector"]),f"{desc.iloc[-1][col]:.1f}%")
    st.markdown("### Sector distribution")
    l,r=st.columns(2)
    with l:
        hist=px.histogram(desc,x=col,nbins=10,marginal="rug",title=f"Distribution of {metric.lower()} ({selected_year})")
        hist.update_layout(xaxis_title="Reported share (%)",yaxis_title="Number of sectors")
        st.plotly_chart(hist,use_container_width=True)
    with r:
        pie_base=sector.copy()
        if total_col:
            top=pie_base.nlargest(5,total_col)[["Sector",total_col]].copy()
            other=pie_base.loc[~pie_base.index.isin(top.index),total_col].sum()
            pie_data=pd.concat([top.rename(columns={total_col:"Establishments"}),pd.DataFrame([{"Sector":"Others","Establishments":other}])],ignore_index=True)
            pie=px.pie(pie_data,names="Sector",values="Establishments",hole=.42,title=f"Share of establishments by sector ({selected_year})")
            st.plotly_chart(pie,use_container_width=True)
        else:
            st.info("Establishment count data is unavailable for this year.")
    st.markdown("### Sector comparison")
    bar_data=desc.copy()
    bar_data["Sector"]=bar_data["Sector"].map(clean_label)
    bar=px.bar(bar_data.sort_values(col),x=col,y="Sector",orientation="h",text=col,title=f"{metric} by sector ({selected_year})")
    bar.update_traces(texttemplate="%{text:.1f}%",textposition="outside",cliponaxis=False)
    bar.update_layout(xaxis_title="Reported share (%)",yaxis_title="",height=520,margin=dict(r=45))
    st.plotly_chart(bar,use_container_width=True)
    st.markdown("### Relationship analysis")
    scatter_x=st.selectbox("X-axis variable",["Number of establishments","Establishments with computers and communication equipment","Establishments with internet access","Establishments with web presence"],index=0)
    scatter_y=st.selectbox("Y-axis variable",["Establishments with computers and communication equipment","Establishments with internet access","Establishments with web presence","Establishments with e-commerce via the internet"],index=1)
    xcol=value_col(raw,scatter_x,selected_year); ycol=value_col(raw,scatter_y,selected_year)
    if xcol and ycol:
        plot_df=sector[["Sector",xcol,ycol]].dropna()
        fig=px.scatter(plot_df,x=xcol,y=ycol,hover_name="Sector",trendline="ols",title=f"{scatter_x} vs. {scatter_y} ({selected_year})")
        fig.update_layout(height=430)
        st.plotly_chart(fig,use_container_width=True)
        corr=plot_df[xcol].corr(plot_df[ycol])
        st.markdown(f'<div class="info-panel"><b>Interpretation</b><br>The Pearson correlation for the selected variables is <b>{corr:.3f}</b>. This describes the strength and direction of a linear association in these sector observations; it does not establish causation.</div>',unsafe_allow_html=True)
    st.markdown("### Descriptive statistics")
    stats=desc[col].describe().rename(index={"count":"Sector count","mean":"Mean","std":"Standard deviation","min":"Minimum","25%":"25th percentile","50%":"Median","75%":"75th percentile","max":"Maximum"}).to_frame("Value")
    st.dataframe(stats.style.format("{:,.2f}"),use_container_width=True)
    st.markdown("### Sector-level data table")
    display=desc.rename(columns={col:f"{metric} ({selected_year})"}).copy()
    display["Sector"]=display["Sector"].map(clean_label)
    st.dataframe(display.reset_index(drop=True),use_container_width=True,hide_index=True)
    st.caption("The pie chart follows the notebook's method: top five sectors by establishment count plus an Others category. The histogram and sector comparison use the selected indicator.")

# ---------- Predictive ----------
elif page == "Predictive Analysis":
    section_header("TREND PROJECTION", "Predictive Analysis", "A linear-regression forecast of average internet access across industry sectors, following the Colab notebook workflow.")
    hist_years=np.array([y for y in years if value_col(raw,"Establishments with internet access",y)])
    hist_values=np.array([sector[value_col(raw,"Establishments with internet access",y)].mean() for y in hist_years],dtype=float)
    valid=np.isfinite(hist_values)
    hist_years,hist_values=hist_years[valid],hist_values[valid]
    if len(hist_years)<2:
        st.error("At least two historical years are required to calculate a trend.")
        st.stop()
    model=LinearRegression().fit(hist_years.reshape(-1,1),hist_values)
    forecast_years=np.arange(2022,2027)
    forecast_values=model.predict(forecast_years.reshape(-1,1))
    forecast_values=np.clip(forecast_values,0,100)
    last=hist_values[-1]; final=forecast_values[-1]
    change=final-last
    a,b,c=st.columns(3)
    with a: metric_card("Latest historical mean",f"{last:.2f}%","2021 sector average")
    with b: metric_card("2026 projected mean",f"{final:.2f}%","Linear trend estimate")
    with c: metric_card("Change vs. 2021",f"{change:+.2f} pp","Percentage-point difference")
    st.markdown("### Historical trend and five-year projection")
    hist_df=pd.DataFrame({"Year":hist_years,"Average internet access (%)":hist_values,"Series":"Historical average"})
    pred_df=pd.DataFrame({"Year":forecast_years,"Average internet access (%)":forecast_values,"Series":"Forecast"})
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist_years,y=hist_values,mode="lines+markers",name="Historical average",line=dict(width=3,color="#234b78")))
    fig.add_trace(go.Scatter(x=forecast_years,y=forecast_values,mode="lines+markers",name="Predicted forecast",line=dict(width=3,dash="dash",color="#d05a55")))
    fig.add_vline(x=2021,line_dash="dot",line_color="gray")
    fig.update_layout(title="Sector average internet access: historical and projected",xaxis_title="Year",yaxis_title="Average internet access (%)",yaxis=dict(range=[0,100]),height=440,legend=dict(orientation="h",y=-.2))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Forecast values")
    forecast_table=pd.DataFrame({"Year":forecast_years,"Predicted average internet access (%)":forecast_values,"Change from 2021 (percentage points)":forecast_values-last})
    st.dataframe(forecast_table.style.format({"Predicted average internet access (%)":"{:.2f}%","Change from 2021 (percentage points)":"{:+.2f}"}),use_container_width=True,hide_index=True)
    st.markdown("### How to interpret this forecast")
    st.markdown("""
    <div class="info-panel">
    <b>Method.</b> The notebook calculates the mean internet-access percentage across individual sectors for each available historical year (2010, 2013, 2015, 2017, 2019 and 2021), then fits a simple Linear Regression model with year as the predictor. The model is extended to 2022–2026.<br><br>
    <b>What it means.</b> The projected values represent an extrapolated sector-average trend, not official future statistics or a guarantee of future performance. Actual outcomes may differ because the model uses only a small number of historical time points and a straight-line trend.<br><br>
    <b>Important limitation.</b> The forecast uses unweighted averages across sectors, matching the notebook. It does not weight sectors by establishment count.
    </div>
    """,unsafe_allow_html=True)
    st.caption("Forecast estimates are recalculated from the uploaded raw dataset when the app runs.")

# ---------- Supervised model ----------
elif page == "Supervised Model":
    section_header("CLASSIFICATION & EVALUATION", "Supervised Model", "Compare supervised classifiers for identifying whether a sector's 2021 internet-access value is above the dataset median.")
    target="Establishments with internet access 2021"
    if target not in ml.columns:
        st.error(f"The structured ML dataset is missing the target column: {target}")
        st.stop()
    data=ml.copy()
    data=data.apply(pd.to_numeric,errors="coerce")
    data=data.dropna(axis=1,how="all")
    data=data.fillna(data.mean(numeric_only=True))
    y=(data[target]>data[target].median()).astype(int)
    feature_cols=[c for c in data.columns if "2021" not in c and c!=target]
    X=data[feature_cols].replace([np.inf,-np.inf],np.nan)
    # Remove columns with no usable variation.
    X=X.loc[:,X.nunique(dropna=True)>1]
    if len(X)<6 or y.nunique()<2:
        st.error("The current data does not contain enough rows or target classes to train classifiers.")
        st.stop()
    n_splits=min(5,int(y.value_counts().min()))
    if n_splits<2:
        st.error("Stratified cross-validation requires at least two observations in each target class.")
        st.stop()
    models={
        "Logistic Regression (L2)":LogisticRegression(C=0.05,penalty="l2",solver="liblinear",random_state=42,max_iter=2000),
        "Gaussian Naive Bayes":GaussianNB(var_smoothing=1e-1),
        "Decision Tree (Pruned)":DecisionTreeClassifier(max_depth=2,min_samples_leaf=2,random_state=42),
        "Random Forest (Robust)":RandomForestClassifier(n_estimators=30,max_depth=2,max_features="log2",random_state=42)
    }
    # Cross-validated, fold-local preprocessing and feature selection to reduce leakage.
    scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
    results=[]
    for name,estimator in models.items():
        pipe=Pipeline([("imputer",SimpleImputer(strategy="mean")),("select",SelectKBest(score_func=f_classif,k=min(10,X.shape[1]))),("scale",StandardScaler()),("model",clone(estimator))])
        scores=cross_validate(pipe,X,y,cv=cv,scoring=scoring,error_score="raise")
        results.append({"Model":name,"Accuracy":scores["test_accuracy"].mean()*100,"Precision":scores["test_precision"].mean()*100,"Recall":scores["test_recall"].mean()*100,"F1-score":scores["test_f1"].mean()*100,"ROC-AUC":scores["test_roc_auc"].mean()})
    result_df=pd.DataFrame(results).sort_values("F1-score",ascending=False)
    best_name=result_df.iloc[0]["Model"]
    best_estimator=models[best_name]
    best_pipe=Pipeline([("imputer",SimpleImputer(strategy="mean")),("select",SelectKBest(score_func=f_classif,k=min(10,X.shape[1]))),("scale",StandardScaler()),("model",clone(best_estimator))])
    # Out-of-fold predictions and probabilities for the selected model.
    from sklearn.model_selection import cross_val_predict
    y_pred=cross_val_predict(best_pipe,X,y,cv=cv,method="predict")
    y_prob=cross_val_predict(best_pipe,X,y,cv=cv,method="predict_proba")[:,1]
    c1,c2,c3,c4=st.columns(4)
    bestrow=result_df.iloc[0]
    with c1: metric_card("Top model",best_name,"Highest mean F1-score")
    with c2: metric_card("Mean accuracy",f"{bestrow['Accuracy']:.1f}%","Stratified cross-validation")
    with c3: metric_card("Mean F1-score",f"{bestrow['F1-score']:.1f}%","Balance of precision and recall")
    with c4: metric_card("Validation folds",str(n_splits),"Stratified K-fold")
    st.markdown("### Model comparison")
    st.dataframe(result_df.style.format({c:"{:.2f}%" for c in ["Accuracy","Precision","Recall","F1-score"]}.format if False else {"Accuracy":"{:.2f}%","Precision":"{:.2f}%","Recall":"{:.2f}%","F1-score":"{:.2f}%","ROC-AUC":"{:.3f}"}),use_container_width=True,hide_index=True)
    metric_chart=result_df.melt(id_vars="Model",value_vars=["Accuracy","Precision","Recall","F1-score"],var_name="Metric",value_name="Score (%)")
    fig=px.bar(metric_chart,x="Model",y="Score (%)",color="Metric",barmode="group",title="Cross-validated model performance")
    fig.update_layout(height=430,xaxis_tickangle=-15)
    st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        st.markdown(f"### Confusion matrix — {best_name}")
        cm=confusion_matrix(y,y_pred,labels=[0,1])
        cmfig=px.imshow(cm,text_auto=True,x=["Predicted low","Predicted high"],y=["Actual low","Actual high"],color_continuous_scale="Blues",aspect="auto")
        cmfig.update_layout(height=360,coloraxis_showscale=False)
        st.plotly_chart(cmfig,use_container_width=True)
        st.caption("Rows are actual classes; columns are predicted classes. Low = at or below median; High = above median.")
    with right:
        st.markdown("### ROC curve — selected model")
        fpr,tpr,_=roc_curve(y,y_prob)
        roc_auc=auc(fpr,tpr)
        rocfig=go.Figure()
        rocfig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"ROC (AUC = {roc_auc:.3f})",line=dict(width=3,color="#2563eb")))
        rocfig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line=dict(dash="dash",color="gray")))
        rocfig.update_layout(xaxis_title="False positive rate",yaxis_title="True positive rate",xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1.02]),height=360,legend=dict(orientation="h",y=-.2))
        st.plotly_chart(rocfig,use_container_width=True)
    st.markdown("### Model and target explanation")
    st.markdown("""
    <div class="info-panel">
    <b>Target variable.</b> Establishments with internet access in 2021 is converted into two classes using its median: values above the median are High Access (1), while values at or below the median are Low Access (0).<br><br>
    <b>Predictors.</b> Features from years other than 2021 are used so that the model does not directly receive same-year 2021 indicator columns as inputs. The top ten features are selected within each validation fold, then missing values are imputed and predictors standardized.<br><br>
    <b>Validation.</b> The comparison uses stratified K-fold cross-validation. Accuracy measures overall correctness; precision measures how often high-access predictions are correct; recall measures how many actual high-access cases are found; F1-score balances precision and recall; ROC-AUC measures ranking separation between the two classes.
    </div>
    """,unsafe_allow_html=True)
    st.markdown("### Cross-validation predictions")
    pred_table=pd.DataFrame({"Sector code":data["Section_Code"] if "Section_Code" in data.columns else np.arange(1,len(data)+1),"Actual class":np.where(y==1,"High Access","Low Access"),"Predicted class":np.where(y_pred==1,"High Access","Low Access"),"Predicted high-access probability":y_prob})
    st.dataframe(pred_table.style.format({"Predicted high-access probability":"{:.3f}"}),use_container_width=True,hide_index=True)
    st.warning("The dataset contains only a small number of sector-level rows. Cross-validation scores are useful for demonstration and comparison, but should not be treated as evidence of production-ready predictive performance.")
