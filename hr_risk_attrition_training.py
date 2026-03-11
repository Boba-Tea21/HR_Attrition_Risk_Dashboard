# -*- coding: utf-8 -*-
"""
HR Attrition Risk Prediction

"""

# ─────────────────────────────────────────
# 1. INSTALL & IMPORT
# ─────────────────────────────────────────
import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ─────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
df_original = pd.read_csv(os.path.join(path, "WA_Fn-UseC_-HR-Employee-Attrition.csv"))

print("Dataset shape:", df_original.shape)
print(df_original['Attrition'].value_counts())

# ─────────────────────────────────────────
# 3. PREPROCESS
# ─────────────────────────────────────────
df = df_original.copy()

# Drop columns with no predictive value
df = df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1)

# Encode target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
df = pd.get_dummies(df, columns=[
    'Department', 'JobRole', 'MaritalStatus',
    'EducationField', 'BusinessTravel', 'Gender', 'OverTime'
])

# ─────────────────────────────────────────
# 4. HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────
majority = df[df.Attrition == 0]
minority = df[df.Attrition == 1]

minority_upsampled = resample(
    minority, replace=True,
    n_samples=len(majority),
    random_state=42
)

df_balanced = pd.concat([majority, minority_upsampled])

# ─────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────
X = df_balanced.drop('Attrition', axis=1)
y = df_balanced['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_pred, y_test))
print("AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# ─────────────────────────────────────────
# 6. SHAP EXPLAINABILITY
# ─────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# ─────────────────────────────────────────
# 7. GENERATE RISK SCORES & EXPORT
# ─────────────────────────────────────────
original_X = df.drop('Attrition', axis=1)
original_X = original_X.reindex(columns=X_train.columns, fill_value=0)

df_original['AttritionRisk'] = model.predict_proba(original_X)[:, 1]
df_original['RiskCategory'] = pd.cut(
    df_original['AttritionRisk'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

df_original.to_csv("hr_complete.csv", index=False)
print("hr_complete.csv exported successfully!")


# ─────────────────────────────────────────
# 8. CORRELATION HEATMAP
# ─────────────────────────────────────────
cols = ['JobSatisfaction', 'OverTime_Yes',
        'MonthlyIncome', 'WorkLifeBalance',
        'AttritionRisk']

corr = df[cols].assign(AttritionRisk=df_original['AttritionRisk']).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0, linewidths=0.5
)
plt.title('Correlation Heatmap — Attrition Risk Drivers')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
