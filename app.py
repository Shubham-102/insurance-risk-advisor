"""
app.py — AI Insurance Risk Advisor
Author: Shubham Maheshwari
Run locally: streamlit run app.py
"""

import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

from src.prediction import load_model_artifacts, get_prediction
from src.llm import get_llm_explanation, get_llm_coverage, FACTOR_LABELS
from src.charts import make_shap_chart, make_gauge

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Insurance Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e8e8; }

[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #2a2f3d; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #e8e8e8; }

.main-header { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #ffffff; line-height: 1.1; margin-bottom: 0.2rem; }
.main-sub { font-size: 1rem; color: #8b92a5; font-weight: 300; letter-spacing: 0.04em; margin-bottom: 2rem; }

.risk-badge { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 2rem; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; }
.risk-high   { background: #3d1515; color: #ff6b6b; border: 1px solid #ff4444; }
.risk-medium { background: #3d2a10; color: #ffaa55; border: 1px solid #ff8800; }
.risk-low    { background: #0f2d1a; color: #55cc88; border: 1px solid #22aa55; }

.metric-card  { background: #161b27; border: 1px solid #2a2f3d; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #ffffff; line-height: 1.1; }
.metric-label { font-size: 0.78rem; color: #8b92a5; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }

.section-title { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #ffffff; border-bottom: 1px solid #2a2f3d; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }

.explanation-box { background: #161b27; border: 1px solid #2a2f3d; border-left: 3px solid #4f8ef7; border-radius: 8px; padding: 1.2rem 1.5rem; font-size: 0.92rem; line-height: 1.7; color: #c8cdd8; white-space: pre-wrap; }
.coverage-box    { background: #161b27; border: 1px solid #2a2f3d; border-left: 3px solid #22aa77; border-radius: 8px; padding: 1.2rem 1.5rem; font-size: 0.92rem; line-height: 1.7; color: #c8cdd8; white-space: pre-wrap; }

.divider { border: none; border-top: 1px solid #2a2f3d; margin: 1.5rem 0; }
.stSlider label   { color: #c8cdd8 !important; font-size: 0.88rem !important; }
.stSelectbox label { color: #c8cdd8 !important; font-size: 0.88rem !important; }

.stButton button { background: linear-gradient(135deg, #4f8ef7, #7b5cf7) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.6rem 2rem !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important; letter-spacing: 0.03em !important; width: 100% !important; transition: opacity 0.2s !important; }
.stButton button:hover { opacity: 0.88 !important; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    load_dotenv()
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    model, label_encoders, features, explainer = load_model_artifacts()
    client = Groq(api_key=api_key)
    return model, label_encoders, features, explainer, client

model, label_encoders, FEATURES, explainer, groq_client = load_all()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.5rem; color: #fff;'>🛡️ Risk Advisor</div>
        <div style='font-size: 0.78rem; color: #8b92a5; margin-top: 0.2rem;'>AI-Powered Insurance Analysis</div>
    </div>
    <hr style='border-color: #2a2f3d; margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("#### Your Profile")
    age      = st.slider("Age", 18, 64, 35)
    sex      = st.selectbox("Sex", ["male", "female"])
    bmi      = st.slider("BMI", 15.0, 53.0, 27.0, step=0.5)
    children = st.slider("Dependents", 0, 5, 0)
    smoker   = st.selectbox("Smoking Status", ["no", "yes"],
                            format_func=lambda x: "Non-smoker" if x == "no" else "Smoker")
    region   = st.selectbox("US Region", ["northeast", "northwest", "southeast", "southwest"],
                            format_func=lambda x: x.capitalize())

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze My Risk →")

    st.markdown("<hr style='border-color: #2a2f3d; margin: 1.5rem 0 0.8rem 0;'>", unsafe_allow_html=True)

    bmi_label = ('Underweight' if bmi < 18.5 else 'Normal weight' if bmi < 25
                 else 'Overweight' if bmi < 30 else 'Obese' if bmi < 35 else 'Severely obese')
    bmi_color = '#55cc88' if bmi < 25 else '#ffaa55' if bmi < 30 else '#ff6b6b'
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#8b92a5;'>BMI Reference</div>
    <div style='font-size:0.92rem; color:{bmi_color}; font-weight:600;'>{bmi:.1f} — {bmi_label}</div>
    <div style='font-size:0.72rem; color:#8b92a5; margin-top:0.4rem;'>
        &lt;18.5 Underweight · 18.5–24.9 Normal<br>25–29.9 Overweight · 30+ Obese
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#5a5f70; line-height:1.6;'>
        Model: XGBoost · R²=0.86<br>
        LLM: Llama 3.3-70b via Groq<br>
        Built by Shubham Maheshwari
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>AI Insurance Risk Advisor</div>
<div class='main-sub'>Machine learning risk assessment · SHAP explainability · AI-powered recommendations</div>
""", unsafe_allow_html=True)

if not analyze:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-value'>86%</div><div class='metric-label'>Model R² Score</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-value'>1,338</div><div class='metric-label'>Training Records</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-value'>16</div><div class='metric-label'>Engineered Features</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#161b27; border:1px solid #2a2f3d; border-radius:12px; padding:2rem; text-align:center;'>
        <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#ffffff; margin-bottom:0.8rem;'>Fill in your profile and click Analyze</div>
        <div style='color:#8b92a5; font-size:0.9rem; line-height:1.8; max-width:480px; margin:0 auto;'>
            The advisor will predict your estimated annual insurance premium,
            identify the key risk factors using SHAP explainability,
            and provide AI-generated plain-English recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    with st.spinner("Running ML model and generating AI recommendations..."):
        charge, segment, top_factors, full_shap = get_prediction(
            age, sex, bmi, children, smoker, region,
            model, label_encoders, FEATURES, explainer
        )
        explanation = get_llm_explanation(age, sex, bmi, children, smoker, region, segment, top_factors, groq_client)
        coverage    = get_llm_coverage(age, sex, bmi, children, smoker, region, segment, charge, groq_client)

    risk_class = {'HIGH': 'risk-high', 'MEDIUM': 'risk-medium', 'LOW': 'risk-low'}[segment]
    risk_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[segment]
    st.markdown(f"<span class='risk-badge {risk_class}'>{risk_emoji} {segment} RISK</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    avg    = 13270
    vs_avg = ((charge - avg) / avg) * 100
    sign   = "+" if vs_avg > 0 else ""
    color  = "#ff6b6b" if vs_avg > 0 else "#55cc88"
    top_feat = FACTOR_LABELS.get(top_factors.iloc[0]['feature'], top_factors.iloc[0]['feature'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>${charge:,.0f}</div><div class='metric-label'>Est. Annual Premium</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>${charge/12:,.0f}</div><div class='metric-label'>Est. Monthly Cost</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{color};'>{sign}{vs_avg:.0f}%</div><div class='metric-label'>vs. US Average</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1rem; padding-top:0.4rem;'>{top_feat.title()}</div><div class='metric-label'>Top Risk Driver</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("<div class='section-title'>Risk Factor Analysis (SHAP)</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#8b92a5; margin-bottom:0.8rem;'>🔴 Red bars push your premium <b>higher</b> &nbsp;·&nbsp; 🟢 Green bars push it <b>lower</b></div>", unsafe_allow_html=True)
        st.plotly_chart(make_shap_chart(top_factors), use_container_width=True)
    with col_right:
        st.markdown("<div class='section-title'>Premium Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(charge), use_container_width=True)
        st.markdown("<div style='font-size:0.82rem; color:#8b92a5; margin-top:0.5rem;'>Top drivers:</div>", unsafe_allow_html=True)
        for _, row in top_factors.head(4).iterrows():
            icon  = "▲" if row['direction'] == 'increases' else "▼"
            color = "#ff6b6b" if row['direction'] == 'increases' else "#55cc88"
            label = FACTOR_LABELS.get(row['feature'], row['feature'])
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:0.3rem 0;
                border-bottom:1px solid #1e2330; font-size:0.83rem;'>
                <span style='color:#c8cdd8;'>{label.title()}</span>
                <span style='color:{color}; font-weight:600;'>{icon} {abs(row['shap_value']):.3f}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_exp, col_cov = st.columns(2)
    with col_exp:
        st.markdown("<div class='section-title'>📋 Risk Explanation</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='explanation-box'>{explanation}</div>", unsafe_allow_html=True)
    with col_cov:
        st.markdown("<div class='section-title'>🏥 Coverage Recommendation</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='coverage-box'>{coverage}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    report_text = f"""INSURANCE RISK ADVISOR REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*50}
CUSTOMER PROFILE
Age: {age} | Sex: {sex} | BMI: {bmi}
Children: {children} | Smoker: {smoker} | Region: {region}

RISK ASSESSMENT
Risk Level: {segment} RISK
Estimated Annual Premium: ${charge:,.2f}
Estimated Monthly Premium: ${charge/12:,.2f}
vs US Average: {sign}{vs_avg:.1f}%

TOP RISK FACTORS (SHAP)
{chr(10).join([f"  {FACTOR_LABELS.get(r['feature'], r['feature'])}: {r['direction']} risk ({r['shap_value']:+.3f})" for _, r in top_factors.head(4).iterrows()])}

RISK EXPLANATION
{explanation}

COVERAGE RECOMMENDATION
{coverage}
{'='*50}
Model: XGBoost | R²=0.86 | MAE=$1,932
LLM: Llama 3.3-70b via Groq
Built by Shubham Maheshwari
"""
    st.download_button(
        label="⬇ Download Full Report",
        data=report_text,
        file_name=f"insurance_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )