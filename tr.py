import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import joblib

# LOAD DATA
df = pd.read_csv("persona_churn_dataset.csv")

# ONE-HOT ENCODE
df = pd.get_dummies(df, columns=["plan_type"], drop_first=False)

X = df.drop("churn", axis=1)
y = df["churn"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# IMBALANCE WEIGHT (ONLY THIS)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

# MODEL
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(y_pred)
print(y_prob)


# METRICS
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# SAVE MODEL
joblib.dump(model, "persona_churn_model.pkl")
print("✅ Model saved")

# BUSINESS OUTPUT
output = X_test.copy()
output["actual_churn"] = y_test.values
output["churn_probability"] = y_prob

def segment(p):
    if p < 0.3:
        return "low risk"
    elif p < 0.7:
        return "medium risk"
    else:
        return "high risk"

def action(seg):
    if seg == "high risk":
        return "call + discount"
    elif seg == "medium risk":
        return "send reminder email"
    else:
        return "upsell premium"

output["risk_segment"] = output["churn_probability"].apply(segment)
output["recommended_action"] = output["risk_segment"].apply(action)

print(output.head())
