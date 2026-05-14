"""
app.py — AI Insurance Risk Advisor
Author: Shubham Maheshwari
Week 4: Streamlit Frontend + Deployment

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect to share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import shap
import os
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Insurance Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e8e8e8;
}

/* Header */
.main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.main-sub {
    font-size: 1rem;
    color: #8b92a5;
    font-weight: 300;
    letter-spacing: 0.04em;
    margin-bottom: 2rem;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 2rem;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.risk-high   { background: #3d1515; color: #ff6b6b; border: 1px solid #ff4444; }
.risk-medium { background: #3d2a10; color: #ffaa55; border: 1px solid #ff8800; }
.risk-low    { background: #0f2d1a; color: #55cc88; border: 1px solid #22aa55; }

/* Metric cards */
.metric-card {
    background: #161b27;
    border: 1px solid #2a2f3d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #ffffff;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #8b92a5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Section headers */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #ffffff;
    border-bottom: 1px solid #2a2f3d;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Explanation box */
.explanation-box {
    background: #161b27;
    border: 1px solid #2a2f3d;
    border-left: 3px solid #4f8ef7;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #c8cdd8;
    white-space: pre-wrap;
}

/* Coverage box */
.coverage-box {
    background: #161b27;
    border: 1px solid #2a2f3d;
    border-left: 3px solid #22aa77;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #c8cdd8;
    white-space: pre-wrap;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #2a2f3d;
    margin: 1.5rem 0;
}

/* Slider labels */
.stSlider label { color: #c8cdd8 !important; font-size: 0.88rem !important; }

/* Select box */
.stSelectbox label { color: #c8cdd8 !important; font-size: 0.88rem !important; }

/* Button */
.stButton button {
    background: linear-gradient(135deg, #4f8ef7, #7b5cf7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.88 !important; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts (cached so they only load once) ───────────────────────────
@st.cache_resource
def load_artifacts():
    load_dotenv()
    model          = joblib.load('insurance_risk_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    with open('features.json') as f:
        features = json.load(f)
    explainer = shap.TreeExplainer(model)
    client    = Groq(api_key=os.getenv('GROQ_API_KEY'))
    return model, label_encoders, features, explainer, client

model, label_encoders, FEATURES, explainer, groq_client = load_artifacts()


# ── Feature engineering (identical to Weeks 2 & 3) ──────────────────────────
def engineer_features(df):
    df = df.copy()
    def bmi_cat(b):
        if b < 18.5: return 'Underweight'
        elif b < 25: return 'Normal'
        elif b < 30: return 'Overweight'
        elif b < 35: return 'Obese'
        else:        return 'Severely_Obese'
    df['bmi_category']    = df['bmi'].apply(bmi_cat)
    df['age_group']       = pd.cut(df['age'], bins=[0,25,35,45,55,100],
                                   labels=['18-25','26-35','36-45','46-55','55+']).astype(str)
    df['is_smoker']       = (df['smoker'] == 'yes').astype(int)
    df['is_obese']        = (df['bmi'] >= 30).astype(int)
    df['smoker_obese']    = df['is_smoker'] * df['is_obese']
    df['smoker_age_risk'] = df['is_smoker'] * (df['age'] >= 40).astype(int)
    df['obese_age_risk']  = df['is_obese']  * (df['age'] >= 40).astype(int)
    df['triple_risk']     = ((df['smoker']=='yes') & (df['bmi']>=30) & (df['age']>=40)).astype(int)
    df['family_size']     = df['children'] + 1
    df['bmi_age']         = df['bmi'] * df['age'] / 1000
    df['bmi_smoker']      = df['bmi'] * df['is_smoker']
    df['is_high_risk']    = 0
    return df


# ── ML prediction + SHAP ────────────────────────────────────────────────────
def get_prediction(age, sex, bmi, children, smoker, region):
    raw = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi, 'children': children,
        'smoker': smoker, 'region': region,
        'charges': 0, 'risk_segment': 'UNKNOWN', 'log_charges': 0
    }])
    raw     = engineer_features(raw)
    X_input = raw[FEATURES].copy()
    for col, le in label_encoders.items():
        X_input[col] = le.transform(X_input[col].astype(str))

    log_pred = model.predict(X_input)[0]
    charge   = float(np.expm1(log_pred))
    segment  = 'HIGH' if charge > 25000 else ('MEDIUM' if charge > 10000 else 'LOW')

    shap_vals = explainer.shap_values(X_input)[0]
    shap_df   = pd.DataFrame({'feature': FEATURES, 'shap_value': shap_vals})
    shap_df['direction'] = shap_df['shap_value'].apply(lambda x: 'increases' if x > 0 else 'decreases')
    shap_df['abs_shap']  = shap_df['shap_value'].abs()
    top_factors = shap_df.nlargest(6, 'abs_shap')

    return charge, segment, top_factors, shap_df


# ── LLM calls ────────────────────────────────────────────────────────────────
FACTOR_LABELS = {
    'is_smoker': 'smoking status', 'bmi_smoker': 'smoking and body weight combined',
    'age': 'age', 'bmi': 'body mass index', 'bmi_age': 'age and weight combined',
    'triple_risk': 'smoking + obesity + age over 40', 'smoker_obese': 'smoking and obesity',
    'smoker_age_risk': 'smoking over age 40', 'children': 'number of dependents',
    'region': 'geographic region', 'is_obese': 'obesity status',
    'obese_age_risk': 'obesity combined with age', 'family_size': 'family size',
    'sex': 'gender', 'age_group': 'age group', 'bmi_category': 'BMI category'
}

SYSTEM_EXPLAIN = """
You are an AI insurance advisor. Explain risk assessments in plain, empathetic English.
Rules: no jargon, warm professional tone, under 150 words, never mention dollar amounts.
Format exactly:
RISK SUMMARY: (1-2 sentences)
KEY FACTORS:
• [factor] — [brief explanation]
• [factor] — [brief explanation]
WHAT YOU CAN DO:
• [action]
• [action]
• [action]
"""

SYSTEM_COVERAGE = """
You are a US insurance coverage advisor. Recommend 2 plans, under 100 words total.
Plans available: HMO, PPO, HDHP+HSA, Comprehensive Plan, Term Life Insurance.
Format exactly:
RECOMMENDED PLANS:
1. [Plan name]: [one sentence why]
2. [Plan name]: [one sentence why]
COVERAGE NOTE: [one key consideration]
"""

def get_llm_explanation(age, sex, bmi, children, smoker, region, segment, top_factors):
    bmi_label = ('underweight' if bmi < 18.5 else 'healthy weight' if bmi < 25
                 else 'overweight' if bmi < 30 else 'obese' if bmi < 35 else 'severely obese')
    factor_lines = '\n'.join([
        f"  - {FACTOR_LABELS.get(row['feature'], row['feature'])} ({row['direction']} expected costs)"
        for _, row in top_factors.head(3).iterrows()
    ])
    prompt = f"""Customer: {age}yo {sex}, BMI {bmi} ({bmi_label}), {'smoker' if smoker=='yes' else 'non-smoker'}, {children} children, {region} US.
Risk: {segment} RISK
Top factors:
{factor_lines}
Explain this assessment."""

    resp = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'system','content':SYSTEM_EXPLAIN},
                  {'role':'user','content':prompt}],
        max_tokens=400, temperature=0.4
    )
    return resp.choices[0].message.content

def get_llm_coverage(age, sex, bmi, children, smoker, region, segment, charge):
    prompt = f"""Customer: {age}yo {sex}, BMI {bmi}, {'smoker' if smoker=='yes' else 'non-smoker'}, {children} dependents, {region} US.
Risk: {segment} RISK. Premium: {'above' if charge > 13270 else 'below'} average.
What coverage do you recommend?"""

    resp = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'system','content':SYSTEM_COVERAGE},
                  {'role':'user','content':prompt}],
        max_tokens=250, temperature=0.3
    )
    return resp.choices[0].message.content


# ── SHAP bar chart ───────────────────────────────────────────────────────────
def make_shap_chart(top_factors):
    df = top_factors.copy().sort_values('shap_value')
    colors = ['#ff6b6b' if v > 0 else '#55cc88' for v in df['shap_value']]
    labels = [FACTOR_LABELS.get(f, f) for f in df['feature']]

    fig = go.Figure(go.Bar(
        x=df['shap_value'],
        y=labels,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in df['shap_value']],
        textposition='outside',
        textfont=dict(color='#8b92a5', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#c8cdd8', size=12),
        margin=dict(l=10, r=60, t=10, b=10),
        height=280,
        xaxis=dict(
            showgrid=True, gridcolor='#2a2f3d', gridwidth=1,
            zeroline=True, zerolinecolor='#4a4f5d', zerolinewidth=1.5,
            tickfont=dict(size=10, color='#8b92a5'),
            title=dict(text='SHAP Value (impact on prediction)', font=dict(size=11, color='#8b92a5'))
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=11))
    )
    return fig


# ── Risk gauge chart ─────────────────────────────────────────────────────────
def make_gauge(charge):
    max_val = 65000
    pct     = min(charge / max_val, 1.0)
    color   = '#ff4444' if pct > 0.38 else ('#ff8800' if pct > 0.15 else '#22aa55')

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=charge,
        number=dict(prefix='$', valueformat=',.0f',
                    font=dict(family='DM Serif Display', size=28, color='#ffffff')),
        gauge=dict(
            axis=dict(range=[0, max_val], tickwidth=1,
                      tickcolor='#2a2f3d', tickfont=dict(color='#8b92a5', size=10),
                      nticks=6),
            bar=dict(color=color, thickness=0.25),
            bgcolor='#161b27',
            borderwidth=0,
            steps=[
                dict(range=[0, 10000],      color='#0f2d1a'),
                dict(range=[10000, 25000],  color='#2a2010'),
                dict(range=[25000, max_val],color='#2d1010'),
            ],
            threshold=dict(line=dict(color='white', width=2), thickness=0.75, value=charge)
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#c8cdd8'),
        margin=dict(l=20, r=20, t=20, b=10),
        height=200
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input form
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.5rem; color: #fff;'>🛡️ Risk Advisor</div>
        <div style='font-size: 0.78rem; color: #8b92a5; margin-top: 0.2rem;'>AI-Powered Insurance Analysis</div>
    </div>
    <hr style='border-color: #2a2f3d; margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("#### Your Profile")

    age      = st.slider("Age", 18, 64, 35, help="Your current age")
    sex      = st.selectbox("Sex", ["male", "female"])
    bmi      = st.slider("BMI", 15.0, 53.0, 27.0, step=0.5,
                         help="Body Mass Index. Normal range: 18.5–24.9")
    children = st.slider("Dependents", 0, 5, 0, help="Number of children covered")
    smoker   = st.selectbox("Smoking Status", ["no", "yes"],
                            format_func=lambda x: "Non-smoker" if x == "no" else "Smoker")
    region   = st.selectbox("US Region",
                            ["northeast", "northwest", "southeast", "southwest"],
                            format_func=lambda x: x.capitalize())

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze My Risk →")

    st.markdown("<hr style='border-color: #2a2f3d; margin: 1.5rem 0 0.8rem 0;'>", unsafe_allow_html=True)

    # BMI reference
    bmi_label = ('Underweight' if bmi < 18.5 else 'Normal weight' if bmi < 25
                 else 'Overweight' if bmi < 30 else 'Obese' if bmi < 35 else 'Severely obese')
    bmi_color = ('#55cc88' if bmi < 25 else '#ffaa55' if bmi < 30 else '#ff6b6b')
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#8b92a5;'>BMI Reference</div>
    <div style='font-size:0.92rem; color:{bmi_color}; font-weight:600;'>{bmi:.1f} — {bmi_label}</div>
    <div style='font-size:0.72rem; color:#8b92a5; margin-top:0.4rem;'>
        &lt;18.5 Underweight · 18.5–24.9 Normal<br>
        25–29.9 Overweight · 30+ Obese
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


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='main-header'>AI Insurance Risk Advisor</div>
<div class='main-sub'>
    Machine learning risk assessment · SHAP explainability · AI-powered recommendations
</div>
""", unsafe_allow_html=True)

# ── Default state (before analysis) ─────────────────────────────────────────
if not analyze:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>86%</div>
            <div class='metric-label'>Model R² Score</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>1,338</div>
            <div class='metric-label'>Training Records</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>16</div>
            <div class='metric-label'>Engineered Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#161b27; border:1px solid #2a2f3d; border-radius:12px; padding:2rem; text-align:center;'>
        <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#ffffff; margin-bottom:0.8rem;'>
            Fill in your profile and click Analyze
        </div>
        <div style='color:#8b92a5; font-size:0.9rem; line-height:1.8; max-width:480px; margin:0 auto;'>
            The advisor will predict your estimated annual insurance premium,
            identify the key risk factors using SHAP explainability,
            and provide AI-generated plain-English recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Analysis results ─────────────────────────────────────────────────────────
else:
    with st.spinner("Running ML model and generating AI recommendations..."):

        # ML + SHAP
        charge, segment, top_factors, full_shap = get_prediction(
            age, sex, bmi, children, smoker, region
        )

        # LLM calls
        explanation = get_llm_explanation(
            age, sex, bmi, children, smoker, region, segment, top_factors
        )
        coverage = get_llm_coverage(
            age, sex, bmi, children, smoker, region, segment, charge
        )

    # ── Risk badge + key metrics ──────────────────────────────────────────
    risk_class = {'HIGH': 'risk-high', 'MEDIUM': 'risk-medium', 'LOW': 'risk-low'}[segment]
    risk_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[segment]

    st.markdown(f"""
    <span class='risk-badge {risk_class}'>{risk_emoji} {segment} RISK</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>${charge:,.0f}</div>
            <div class='metric-label'>Est. Annual Premium</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        monthly = charge / 12
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>${monthly:,.0f}</div>
            <div class='metric-label'>Est. Monthly Cost</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        avg    = 13270
        vs_avg = ((charge - avg) / avg) * 100
        sign   = "+" if vs_avg > 0 else ""
        color  = "#ff6b6b" if vs_avg > 0 else "#55cc88"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{color};'>{sign}{vs_avg:.0f}%</div>
            <div class='metric-label'>vs. US Average</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        top_feat = FACTOR_LABELS.get(top_factors.iloc[0]['feature'], top_factors.iloc[0]['feature'])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='font-size:1rem; padding-top:0.4rem;'>{top_feat.title()}</div>
            <div class='metric-label'>Top Risk Driver</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Two column layout: chart + gauge ─────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-title'>Risk Factor Analysis (SHAP)</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.8rem; color:#8b92a5; margin-bottom:0.8rem;'>
            🔴 Red bars push your premium <b>higher</b> &nbsp;·&nbsp;
            🟢 Green bars push it <b>lower</b>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(make_shap_chart(top_factors), use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Premium Gauge</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_gauge(charge), use_container_width=True)

        # Risk factor summary list
        st.markdown("<div style='font-size:0.82rem; color:#8b92a5; margin-top:0.5rem;'>Top drivers:</div>",
                    unsafe_allow_html=True)
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

    # ── LLM outputs ──────────────────────────────────────────────────────
    col_exp, col_cov = st.columns(2)

    with col_exp:
        st.markdown("<div class='section-title'>📋 Risk Explanation</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='explanation-box'>{explanation}</div>",
                    unsafe_allow_html=True)

    with col_cov:
        st.markdown("<div class='section-title'>🏥 Coverage Recommendation</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='coverage-box'>{coverage}</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Download report ───────────────────────────────────────────────────
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
vs US Average: {'+' if vs_avg > 0 else ''}{vs_avg:.1f}%

TOP RISK FACTORS (SHAP)
{chr(10).join([f"  {FACTOR_LABELS.get(r['feature'], r['feature'])}: {r['direction']} risk ({r['shap_value']:+.3f})" for _, r in top_factors.head(4).iterrows()])}

RISK EXPLANATION
{explanation}

COVERAGE RECOMMENDATION
{coverage}

{'='*50}
Model: XGBoost | R²=0.86 | MAE=$1,932
LLM: Llama 3.3-70b via Groq (free)
Built by Shubham Maheshwari
"""
    st.download_button(
        label="⬇ Download Full Report",
        data=report_text,
        file_name=f"insurance_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )
