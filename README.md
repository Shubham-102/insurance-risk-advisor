# 🛡️ AI Insurance Risk Advisor

An AI-powered insurance risk assessment tool built with XGBoost, SHAP explainability, and Llama 3.3-70b via Groq.

## 🚀 Live Demo
<!-- Add your Streamlit Cloud URL here once deployed -->

## 📌 What It Does
- Predicts your estimated annual insurance premium using a trained XGBoost model
- Explains the key risk factors driving your premium using SHAP values
- Generates plain-English risk explanations and coverage recommendations via LLM

## 🧰 Tech Stack
| Layer | Tool |
|-------|------|
| Frontend | Streamlit |
| ML Model | XGBoost (R² = 0.86) |
| Explainability | SHAP |
| LLM | Llama 3.3-70b via Groq |
| Data | Health Insurance Dataset (1,338 records) |

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Shubham-102/insurance-risk-advisor.git
cd insurance-risk-advisor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the root:

GROQ_API_KEY=your_key_here

**4. Run the app**
```bash
streamlit run app.py
```

## 📁 Project Structure
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── features.json           # Feature list used by the model
├── model_metrics.json      # Model evaluation metrics
├── data/
│   └── health_insurance_with_segments.csv
└── models/                 # .pkl files (gitignored — see below)

## 🔒 Model Files
The `.pkl` model files are not committed to this repo.
To regenerate them, run the training notebook:
<!-- Add link to notebook once added -->

## 👤 Author
Built by Shubham Maheshwari