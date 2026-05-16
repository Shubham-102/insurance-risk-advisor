FACTOR_LABELS = {
    'is_smoker': 'smoking status',
    'bmi_smoker': 'smoking and body weight combined',
    'age': 'age',
    'bmi': 'body mass index',
    'bmi_age': 'age and weight combined',
    'triple_risk': 'smoking + obesity + age over 40',
    'smoker_obese': 'smoking and obesity',
    'smoker_age_risk': 'smoking over age 40',
    'children': 'number of dependents',
    'region': 'geographic region',
    'is_obese': 'obesity status',
    'obese_age_risk': 'obesity combined with age',
    'family_size': 'family size',
    'sex': 'gender',
    'age_group': 'age group',
    'bmi_category': 'BMI category'
}

SYSTEM_EXPLAIN = """
You are an AI insurance advisor. Explain risk assessments in plain, empathetic English.
Rules: no jargon, warm professional tone, under 150 words, never mention dollar amounts.
Format exactly:
RISK SUMMARY: (1-2 sentences)
KEY FACTORS:
- [factor] — [brief explanation]
- [factor] — [brief explanation]
WHAT YOU CAN DO:
- [action]
- [action]
- [action]
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


def get_llm_explanation(age, sex, bmi, children, smoker, region, segment, top_factors, groq_client):
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
        messages=[{'role': 'system', 'content': SYSTEM_EXPLAIN},
                  {'role': 'user',   'content': prompt}],
        max_tokens=400, temperature=0.4
    )
    return resp.choices[0].message.content


def get_llm_coverage(age, sex, bmi, children, smoker, region, segment, charge, groq_client):
    prompt = f"""Customer: {age}yo {sex}, BMI {bmi}, {'smoker' if smoker=='yes' else 'non-smoker'}, {children} dependents, {region} US.
Risk: {segment} RISK. Premium: {'above' if charge > 13270 else 'below'} average.
What coverage do you recommend?"""

    resp = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'system', 'content': SYSTEM_COVERAGE},
                  {'role': 'user',   'content': prompt}],
        max_tokens=250, temperature=0.3
    )
    return resp.choices[0].message.content