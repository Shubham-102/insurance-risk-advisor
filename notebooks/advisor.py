"""
advisor.py — Core AI Insurance Advisor Pipeline
Author: Shubham Maheshwari

Usage:
    from advisor import run_full_advisor
    report = run_full_advisor(age=35, sex="male", bmi=28, children=1, smoker="no", region="northeast")
"""

import os, json, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from groq import Groq
import shap
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_model          = joblib.load("insurance_risk_model.pkl")
_label_encoders = joblib.load("label_encoders.pkl")
_explainer      = shap.TreeExplainer(_model)
with open("features.json") as f:
    _FEATURES = json.load(f)

SYSTEM_PROMPT = """
You are an AI insurance advisor. Explain risk assessments in plain, empathetic English.
Rules: no jargon, warm tone, under 150 words, never mention dollar amounts.
Format:
  RISK SUMMARY: (1-2 sentences)
  KEY FACTORS: (2-3 bullets)
  WHAT YOU CAN DO: (2-3 recommendations)
"""

COVERAGE_PROMPT = """
You are a US insurance coverage advisor. Recommend 2 plans max, under 100 words.
Plans: HMO, PPO, HDHP+HSA, Comprehensive, Term Life.
Format:
  RECOMMENDED PLANS:
  1. [Plan]: [why]
  2. [Plan]: [why]
  COVERAGE NOTE: [one key consideration]
"""

FACTOR_LABELS = {
    "is_smoker": "smoking status", "bmi_smoker": "smoking and body weight combined",
    "age": "age", "bmi": "body mass index", "bmi_age": "age and weight combined",
    "triple_risk": "smoking + obesity + age over 40", "smoker_obese": "smoking and obesity",
    "smoker_age_risk": "smoking over age 40", "children": "number of dependents",
    "region": "geographic region", "is_obese": "obesity status"
}

def _engineer(df):
    df = df.copy()
    def bmi_cat(b):
        if b<18.5: return "Underweight"
        elif b<25: return "Normal"
        elif b<30: return "Overweight"
        elif b<35: return "Obese"
        else:      return "Severely_Obese"
    df["bmi_category"]    = df["bmi"].apply(bmi_cat)
    df["age_group"]       = pd.cut(df["age"], bins=[0,25,35,45,55,100],
                                   labels=["18-25","26-35","36-45","46-55","55+"]).astype(str)
    df["is_smoker"]       = (df["smoker"]=="yes").astype(int)
    df["is_obese"]        = (df["bmi"]>=30).astype(int)
    df["smoker_obese"]    = df["is_smoker"] * df["is_obese"]
    df["smoker_age_risk"] = df["is_smoker"] * (df["age"]>=40).astype(int)
    df["obese_age_risk"]  = df["is_obese"]  * (df["age"]>=40).astype(int)
    df["triple_risk"]     = ((df["smoker"]=="yes")&(df["bmi"]>=30)&(df["age"]>=40)).astype(int)
    df["family_size"]     = df["children"] + 1
    df["bmi_age"]         = df["bmi"] * df["age"] / 1000
    df["bmi_smoker"]      = df["bmi"] * df["is_smoker"]
    df["is_high_risk"]    = 0
    return df

def run_full_advisor(age, sex, bmi, children, smoker, region):
    # ML prediction
    raw = pd.DataFrame([{"age":age,"sex":sex,"bmi":bmi,"children":children,
                         "smoker":smoker,"region":region,
                         "charges":0,"risk_segment":"UNKNOWN","log_charges":0}])
    raw = _engineer(raw)
    X   = raw[_FEATURES].copy()
    for col, le in _label_encoders.items():
        X[col] = le.transform(X[col].astype(str))
    charge  = float(np.expm1(_model.predict(X)[0]))
    segment = "HIGH" if charge>25000 else ("MEDIUM" if charge>10000 else "LOW")

    # SHAP
    sv  = _explainer.shap_values(X)[0]
    sf  = pd.DataFrame({"feature":_FEATURES,"shap_value":sv})
    sf["direction"] = sf["shap_value"].apply(lambda x: "increases" if x>0 else "decreases")
    sf["abs"]       = sf["shap_value"].abs()
    factors = sf.nlargest(4,"abs").to_dict("records")

    bmi_label = ("underweight" if bmi<18.5 else "healthy weight" if bmi<25
                 else "overweight" if bmi<30 else "obese" if bmi<35 else "severely obese")
    factor_lines = "\n".join(
        f"  - {FACTOR_LABELS.get(f['feature'],f['feature'])} ({f['direction']} costs)"
        for f in factors[:3]
    )
    exp_prompt = (f"Customer: {age}yo {sex}, BMI {bmi} ({bmi_label}), "
                  f"{'smoker' if smoker=='yes' else 'non-smoker'}, "
                  f"{children} children, {region} US.\nRisk: {segment} RISK\n"
                  f"Factors:\n{factor_lines}")
    cov_prompt  = (f"Customer: {age}yo {sex}, BMI {bmi}, {smoker} smoker, "
                   f"{children} dependents, {region}, {segment} RISK.")

    explanation = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":exp_prompt}],
        max_tokens=400, temperature=0.4
    ).choices[0].message.content

    coverage = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":COVERAGE_PROMPT},{"role":"user","content":cov_prompt}],
        max_tokens=250, temperature=0.3
    ).choices[0].message.content

    return {
        "timestamp":               datetime.now().isoformat(),
        "customer_input":          {"age":age,"sex":sex,"bmi":round(bmi,1),
                                    "children":children,"smoker":smoker,"region":region},
        "predicted_annual_charge": round(charge, 2),
        "risk_segment":            segment,
        "top_risk_factors":        [{"feature":f["feature"],"direction":f["direction"],
                                     "impact":round(float(f["shap_value"]),4)} for f in factors],
        "risk_explanation":        explanation,
        "coverage_recommendation": coverage
    }
