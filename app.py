from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib

# --------------------------------------------------
# Load trained model (server start hote hi)
# --------------------------------------------------
model = joblib.load("persona_churn_model.pkl")

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# Input schema
# FIXED: Pehle koi validation nahi tha — koi bhi negative ya galat value accept ho jaati thi
# GALAT: tenure: -50, complaints: -100, payment_missed: 5, plan_type: "xyz" sab chalti thi
# SAHI: Ab Field() se range check lagaya hai — galat input pe clear error milega, garbage prediction nahi
# --------------------------------------------------
class Customer(BaseModel):
    tenure: int = Field(ge=0, description="Months customer has been active")
    last_active_days: int = Field(ge=0, description="Days since last activity")
    usage_frequency: int = Field(ge=0, description="Usage per week")
    session_duration: int = Field(ge=0, description="Session duration in minutes")
    complaints: int = Field(ge=0, description="Number of complaints")
    monthly_charges: float = Field(gt=0, description="Monthly charges in rupees")
    payment_missed: int = Field(ge=0, le=1, description="0 or 1 only")
    products_used: int = Field(ge=1, description="At least 1 product")
    plan_type: Literal["basic", "standard", "premium"]  # FIXED: pehle str tha — koi bhi value chalti thi, ab sirf ye 3 chalegi

# --------------------------------------------------
# Business logic helpers
# --------------------------------------------------
# FIXED: Pehle thresholds 0.2 / 0.6 the jo GALAT the
# Kyunki tr.py (training) mein 0.3 / 0.7 use kiya tha
# Isse production mein alag results aa rahe the vs training evaluation
# Ab SAHI hai: 0.3 / 0.7 — tr.py ke saath match karta hai
def segment(prob):
    if prob < 0.3:
        return "low risk"
    elif prob < 0.7:
        return "medium risk"
    else:
        return "high risk"

def recommend(risk):
    if risk == "low risk":
        return "upsell premium"
    elif risk == "medium risk":
        return "send reminder email"
    else:
        return "discount + call"

# --------------------------------------------------
# Home endpoint
# --------------------------------------------------
@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(customer: Customer):

    # 1. Input → DataFrame
    df = pd.DataFrame([customer.dict()])

    # 2. One-hot encode plan_type
    df = pd.get_dummies(df, columns=["plan_type"], drop_first=False)

    # 3. Align columns with training
    expected_cols = model.feature_names_in_
    df = df.reindex(columns=expected_cols, fill_value=0)

    # 4. Predict probability
    prob = model.predict_proba(df)[0][1]

    # 5. Risk + action
    risk = segment(prob)
    action = recommend(risk)

    # 6. Business override (revenue protection)
    if (
        risk == "low risk"
        and customer.monthly_charges > 900
        and customer.products_used == 1
    ):
        risk = "medium risk"
        action = "send reminder email"

    return {
        "churn_probability": round(float(prob), 4),
        "risk": risk,
        "recommended_action": action
    }

# --------------------------------------------------
# SIMPLE UI ENDPOINT
# --------------------------------------------------
# --------------------------------------------------
# UI ENDPOINT (WITH PREMIUM MODERN DESIGN & EXPLICIT LABELS)
# FIXED: Pehle inputs ke upar unka naam (labels) nahi tha, sirf placeholders the.
# GALAT: Value type karne ke baad placeholder gayab ho jata tha aur pata nahi chalta tha kis box mein kya daala hai.
# SAHI: Ab har box ke upar clear labels laga diye hain aur design ko modern dark-themed 2-column grid bana diya hai.
# --------------------------------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
        <title>Customer Churn Predictor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                min-height: 100vh; 
                background: linear-gradient(135deg, #0f172a, #1e293b); 
                color: #f8fafc;
                display: flex; 
                justify-content: center; 
                align-items: flex-start;
                padding: 40px 20px;
                margin: 0;
            }
            .box { 
                background: rgba(30, 41, 59, 0.7); 
                backdrop-filter: blur(10px);
                padding: 30px; 
                border-radius: 16px; 
                width: 100%;
                max-width: 750px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            h1 {
                text-align: center;
                font-size: 28px;
                font-weight: 700;
                margin-top: 0;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .subtitle {
                text-align: center;
                color: #94a3b8;
                margin-bottom: 25px;
                font-size: 14px;
            }
            h2 {
                font-size: 18px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                padding-bottom: 8px;
                margin-bottom: 20px;
                color: #3b82f6;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 25px;
            }
            .input-group {
                display: flex;
                flex-direction: column;
            }
            label {
                font-size: 13px;
                font-weight: 600;
                color: #94a3b8;
                margin-bottom: 6px;
            }
            input, select { 
                padding: 10px 12px; 
                border-radius: 8px;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #f8fafc;
                font-size: 14px;
                transition: all 0.3s ease;
                outline: none;
            }
            input:focus, select:focus {
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
                background: rgba(15, 23, 42, 0.8);
            }
            button { 
                padding: 12px; 
                width: 100%; 
                background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                color: white; 
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease, opacity 0.2s ease;
            }
            button:hover {
                opacity: 0.95;
            }
            button:active {
                transform: scale(0.98);
            }
            h3 {
                font-size: 18px;
                margin-top: 25px;
                margin-bottom: 10px;
                color: #10b981;
            }
            pre { 
                background: rgba(15, 23, 42, 0.8); 
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 8px;
                padding: 15px; 
                color: #34d399;
                font-family: 'Courier New', Courier, monospace;
                font-size: 14px;
                line-height: 1.5;
                white-space: pre-wrap;
                margin: 0;
            }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>Customer Churn Prediction System</h1>
            <div class="subtitle">Enter customer behavior details to analyze churn risk</div>
            
            <h2>Customer Details</h2>
            <div class="grid">
                <div class="input-group">
                    <label for="tenure">Tenure (months)</label>
                    <input id="tenure" type="number" placeholder="e.g. 12">
                </div>
                <div class="input-group">
                    <label for="last_active_days">Last Active Days</label>
                    <input id="last_active_days" type="number" placeholder="e.g. 5">
                </div>
                <div class="input-group">
                    <label for="usage_frequency">Usage Frequency (per week)</label>
                    <input id="usage_frequency" type="number" placeholder="e.g. 3">
                </div>
                <div class="input-group">
                    <label for="session_duration">Session Duration (minutes)</label>
                    <input id="session_duration" type="number" placeholder="e.g. 45">
                </div>
                <div class="input-group">
                    <label for="complaints">Complaints Count</label>
                    <input id="complaints" type="number" placeholder="e.g. 0">
                </div>
                <div class="input-group">
                    <label for="monthly_charges">Monthly Charges (₹)</label>
                    <input id="monthly_charges" type="number" placeholder="e.g. 600">
                </div>
                <div class="input-group">
                    <label for="payment_missed">Payment Missed (1 = Yes, 0 = No)</label>
                    <input id="payment_missed" type="number" placeholder="e.g. 0">
                </div>
                <div class="input-group">
                    <label for="products_used">Products Used Count</label>
                    <input id="products_used" type="number" placeholder="e.g. 2">
                </div>
                <div class="input-group" style="grid-column: 1 / -1;">
                    <label for="plan_type">Plan Type</label>
                    <select id="plan_type">
                        <option value="basic">Basic</option>
                        <option value="standard">Standard</option>
                        <option value="premium">Premium</option>
                    </select>
                </div>
            </div>

            <button onclick="predict()">Analyze Churn Risk</button>

            <h3>Result Analysis</h3>
            <pre id="result">Click button to generate prediction...</pre>
        </div>

        <script>
            // FIXED: Pehle koi error handling nahi tha
            // GALAT: Agar API validation error deta tha (jaise payment_missed=3) toh "undefined" dikhta tha
            // SAHI: Ab error hone pe clear message dikhega ki kya galat hai
            function predict() {
                fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        tenure: Number(tenure.value),
                        last_active_days: Number(last_active_days.value),
                        usage_frequency: Number(usage_frequency.value),
                        session_duration: Number(session_duration.value),
                        complaints: Number(complaints.value),
                        monthly_charges: Number(monthly_charges.value),
                        payment_missed: Number(payment_missed.value),
                        products_used: Number(products_used.value),
                        plan_type: plan_type.value
                    })
                })
                .then(res => res.json().then(data => ({status: res.status, data: data})))
                .then(({status, data}) => {
                    if (status !== 200) {
                        // Validation error aaya — user ko clear message dikhao
                        let errors = data.detail.map(e => e.loc[1] + ": " + e.msg).join("\\n");
                        document.getElementById("result").innerText = "❌ Input Error:\\n" + errors;
                        return;
                    }
                    document.getElementById("result").innerText =
`Customer Summary
----------------
Tenure           : ${tenure.value} months
Last Active Days : ${last_active_days.value}
Usage Frequency  : ${usage_frequency.value}/week
Session Duration : ${session_duration.value} mins
Complaints       : ${complaints.value}
Monthly Charges  : ₹${monthly_charges.value}
Payment Missed   : ${payment_missed.value == 1 ? "Yes" : "No"}
Products Used    : ${products_used.value}
Plan Type        : ${plan_type.value}

Prediction
-----------
Churn Probability : ${data.churn_probability}
Risk Level        : ${data.risk.toUpperCase()}
Recommended Action: ${data.recommended_action.toUpperCase()}`;
                });
            }
        </script>
    </body>
    </html>
    """

