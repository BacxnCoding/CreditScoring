import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import joblib

# Load data
def load_data():
    data_dir = "./project-root/data"
    applicants = pd.read_csv(os.path.join(data_dir, "applicants.csv"))
    loans = pd.read_csv(os.path.join(data_dir, "loans.csv"))
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    vat = pd.read_csv(os.path.join(data_dir, "vat_invoices.csv"))
    bureau = pd.read_csv(os.path.join(data_dir, "bureau.csv"))

    # Merge data
    df = loans.merge(applicants, on="firm_id", how="left") \
        .merge(transactions.groupby("firm_id").agg({"inflow": "mean", "outflow": "mean", "balance": "mean"}), on="firm_id", how="left") \
        .merge(vat.groupby("firm_id").agg({"sales": "sum", "purchases": "sum", "tax_paid": "sum"}), on="firm_id", how="left") \
        .merge(bureau, on="firm_id", how="left")

    return df

# Feature Engineering
def feature_engineering(df):
    df["avg_net_cf"] = df["inflow"] - df["outflow"]
    df["cf_vol"] = df["balance"].std()
    df["sales_growth"] = df["sales"].pct_change().fillna(0)
    df["purchase_ratio"] = df["purchases"] / (df["sales"] + 1)
    df["debt_service_ratio"] = df["inflow"] / (df["amount"] + 1)
    df["tax_ratio"] = df["tax_paid"] / (df["sales"] + 1)
    
    # Drop columns not needed for modeling
    df = df.drop(columns=["firm_id", "sector", "region", "ownership", "app_date", "default_date"])

    # Clean any missing values or negative values
    df = df.fillna(0)
    # Apply the transformation only to numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].clip(lower=0)

    return df

# Train Models
def train_models(df):
    # Define features and target
    X = df.select_dtypes(include=[np.number]).drop(columns=["outcome"])  # Drop 'outcome' as it's the target
    y = df["outcome"]

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # StandardScaler: Fitting and transforming training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform test data using the trained scaler
    
    # Logistic Regression Model
    logit_model = LogisticRegression(max_iter=1000, random_state=42)
    logit_model.fit(X_train_scaled, y_train)
    y_pred_logit = logit_model.predict_proba(X_test_scaled)[:, 1]
    logit_auc = roc_auc_score(y_test, y_pred_logit)

    # LightGBM Model
    gbm_model = lgb.LGBMClassifier(random_state=42)
    gbm_model.fit(X_train_scaled, y_train)
    y_pred_gbm = gbm_model.predict_proba(X_test_scaled)[:, 1]
    gbm_auc = roc_auc_score(y_test, y_pred_gbm)

    # Metrics and Model Evaluation
    print(f"Logistic Regression AUC: {logit_auc:.4f}")
    print(f"LightGBM AUC: {gbm_auc:.4f}")

    # Save models and scaler
    joblib.dump(logit_model, "./project-root/models/baseline_logit_model.pkl")
    joblib.dump(gbm_model, "./project-root/models/gbm_model.pkl")
    joblib.dump(scaler, './project-root/models/scaler.pkl')  # Save the scaler
    
    return logit_model, gbm_model, X_test, y_test, y_pred_logit, y_pred_gbm

# SHAP Reason Codes (Logit and GBM)
def generate_shap_explainer(model, X_train, model_type="logit"):
    if model_type == "logit":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return shap_values, explainer

# Model Performance Plots
def plot_roc_curve(y_test, y_pred_logit, y_pred_gbm):
    fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_logit)
    fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_pred_gbm)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr_logit, tpr_logit, label="Logistic Regression", color="blue")
    plt.plot(fpr_gbm, tpr_gbm, label="LightGBM", color="green")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall(y_test, y_pred_logit, y_pred_gbm):
    precision_logit, recall_logit, _ = precision_recall_curve(y_test, y_pred_logit)
    precision_gbm, recall_gbm, _ = precision_recall_curve(y_test, y_pred_gbm)
    plt.figure(figsize=(10, 7))
    plt.plot(recall_logit, precision_logit, label="Logistic Regression", color="blue")
    plt.plot(recall_gbm, precision_gbm, label="LightGBM", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()

def save_model_card(logit_auc, gbm_auc):
    model_card = f"""
    ## Model Summary
    - Logistic Regression AUC: {logit_auc:.4f}
    - LightGBM AUC: {gbm_auc:.4f}

    ### Logistic Regression
    - Linear Model trained using Logistic Regression.
    - AUC: {logit_auc:.4f}

    ### LightGBM
    - Boosting Model trained using LightGBM.
    - AUC: {gbm_auc:.4f}
    """
    with open("./project-root/models/model_card.pdf", "w") as f:
        f.write(model_card)

def main():
    df = load_data()
    df = feature_engineering(df)
    
    # Train models
    logit_model, gbm_model, X_test, y_test, y_pred_logit, y_pred_gbm = train_models(df)
    
    # Plot and save performance
    plot_roc_curve(y_test, y_pred_logit, y_pred_gbm)
    plot_precision_recall(y_test, y_pred_logit, y_pred_gbm)
    
    # SHAP reason codes
    shap_values_logit, explainer_logit = generate_shap_explainer(logit_model, X_test, model_type="logit")
    shap_values_gbm, explainer_gbm = generate_shap_explainer(gbm_model, X_test, model_type="gbm")
    
    # Save SHAP plots (Optional)
    shap.summary_plot(shap_values_logit, X_test, plot_type="bar", show=False)
    plt.savefig('./project-root/models/logit_shap_summary.png')
    shap.summary_plot(shap_values_gbm, X_test, plot_type="bar", show=False)
    plt.savefig('./project-root/models/gbm_shap_summary.png')

    # Save model card
    save_model_card(roc_auc_score(y_test, y_pred_logit), roc_auc_score(y_test, y_pred_gbm))

    print("Training complete. Models and metrics saved.")

if __name__ == "__main__":
    main()
