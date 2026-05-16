import pandas as pd
import numpy as np
import shap
import joblib
import json

def load_model_artifacts():
    model = joblib.load('models/insurance_risk_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    with open('config/features.json') as f:
        features = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, label_encoders, features, explainer


def engineer_features(df):
    df = df.copy()

    def bmi_cat(b):
        if b < 18.5: return 'Underweight'
        elif b < 25: return 'Normal'
        elif b < 30: return 'Overweight'
        elif b < 35: return 'Obese'
        else: return 'Severely_Obese'

    df['bmi_category'] = df['bmi'].apply(bmi_cat)
    df['age_group'] = pd.cut(df['age'], bins=[0,25,35,45,55,100],
                             labels=['18-25','26-35','36-45','46-55','55+']).astype(str)
    df['is_smoker']      = (df['smoker'] == 'yes').astype(int)
    df['is_obese']       = (df['bmi'] >= 30).astype(int)
    df['smoker_obese']   = df['is_smoker'] * df['is_obese']
    df['smoker_age_risk']= df['is_smoker'] * (df['age'] >= 40).astype(int)
    df['obese_age_risk'] = df['is_obese']  * (df['age'] >= 40).astype(int)
    df['triple_risk']    = ((df['smoker']=='yes') & (df['bmi']>=30) & (df['age']>=40)).astype(int)
    df['family_size']    = df['children'] + 1
    df['bmi_age']        = df['bmi'] * df['age'] / 1000
    df['bmi_smoker']     = df['bmi'] * df['is_smoker']
    df['is_high_risk']   = 0
    return df


def get_prediction(age, sex, bmi, children, smoker, region, model, label_encoders, features, explainer):
    raw = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi, 'children': children,
        'smoker': smoker, 'region': region,
        'charges': 0, 'risk_segment': 'UNKNOWN', 'log_charges': 0
    }])
    raw = engineer_features(raw)
    X_input = raw[features].copy()

    for col, le in label_encoders.items():
        X_input[col] = le.transform(X_input[col].astype(str))

    log_pred = model.predict(X_input)[0]
    charge   = float(np.expm1(log_pred))
    segment  = 'HIGH' if charge > 25000 else ('MEDIUM' if charge > 10000 else 'LOW')

    shap_vals = explainer.shap_values(X_input)[0]
    shap_df   = pd.DataFrame({'feature': features, 'shap_value': shap_vals})
    shap_df['direction'] = shap_df['shap_value'].apply(lambda x: 'increases' if x > 0 else 'decreases')
    shap_df['abs_shap']  = shap_df['shap_value'].abs()
    top_factors = shap_df.nlargest(6, 'abs_shap')

    return charge, segment, top_factors, shap_df