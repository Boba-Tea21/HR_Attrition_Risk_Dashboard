# HR_Attrition_Risk_Dashboard
"End-to-end HR attrition risk prediction using XGBoost and Power BI"

An end-to-end machine learning and business intelligence project that predicts employee attrition risk, uncovers the key drivers behind it, and visualizes actionable insights through an interactive Power BI dashboard. 

Dataset
Source: [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1,470 employee records
- 35 features covering demographics, compensation, job satisfaction, performance, and work history
- Target variable: Attrition (Yes/No)

---

Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Data preprocessing, feature engineering, modeling |
| XGBoost | Gradient boosting model for attrition risk prediction |
| SHAP | Model explainability — understanding *why* predictions are made |
| Pandas / NumPy | Data manipulation and transformation |
| Seaborn / Matplotlib | Correlation heatmap visualization |
| Scikit-learn | Train/test split, class balancing, evaluation metrics |
| Power BI | Interactive business intelligence dashboard |

---

Model performance
| Metric | Score |
|--------|-------|
| Accuracy | 93% |
| AUC Score | **0.98** |
| Precision (Attrition = Yes) | 94% |
| Recall (Attrition = Yes) | 91% |
| F1 Score | 93% |

An AUC of 0.98 means the model can almost perfectly distinguish between employees who will leave and those who will stay. The model was trained on a class-balanced dataset using upsampling to hangle natural imablance between employees who stay (~84%) and those who leave (~16%). 

Key Insights

Analysis reveals that attrition risk is driven by a combination of workload, compensation, and engagement factors. Overtime is the strongest element predictor of attrition, followed by low montly income. Notetably, top performers carry higher attrition risk than average performers. Together these findings point to three priority retention levers: competitive compensation, overtime management, and targeted engagement programs for high performers. 
