from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import HTMLResponse
import numpy as np
import shap
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import os
from fastapi.staticfiles import StaticFiles

# Ensure correct path to the models folder
model_dir = os.path.join(os.path.dirname(__file__), "../models")

# Load the models and the scaler (no pipeline assumed here)
logit_model = joblib.load(os.path.join(model_dir, "baseline_logit_model.pkl"))
gbm_model = joblib.load(os.path.join(model_dir, "gbm_model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))  # Load the scaler used during training

# Initialize the FastAPI app
app = FastAPI()

# Explicitly resolve the path to the UI folder
ui_path = os.path.join(os.path.dirname(__file__), "../ui")

# Serve the UI page
@app.get("/")
def read_ui():
    with open(os.path.join(ui_path, "upload_ui.html"), "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Serve static files (for any future images, JS, etc.)
app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

# Define the input structure using Pydantic
class InputData(BaseModel):
    avg_inflow: float
    avg_outflow: float
    avg_net_cf: float
    std_net_cf: float
    avg_balance: float
    min_balance: float
    max_drawdown: float
    mom_net_cf_mean: float
    mom_net_cf_std: float
    mom_net_cf_p25: float
    mom_net_cf_p75: float
    avg_sales: float
    avg_purchases: float
    avg_va: float
    tax_paid_mean: float
    sales_std: float
    purchases_std: float
    sales_mom_mean: float
    purchases_mom_mean: float

# Define the /score endpoint to return credit score and grade
@app.post("/score")
def score(data: InputData):
    # Prepare the input data as an array (similar to the features used in training)
    X_input = np.array([[data.avg_inflow, data.avg_outflow, data.avg_net_cf, data.std_net_cf, 
                         data.avg_balance, data.min_balance, data.max_drawdown, data.mom_net_cf_mean, 
                         data.mom_net_cf_std, data.mom_net_cf_p25, data.mom_net_cf_p75, 
                         data.avg_sales, data.avg_purchases, data.avg_va, data.tax_paid_mean, 
                         data.sales_std, data.purchases_std, data.sales_mom_mean, 
                         data.purchases_mom_mean]])

    # Apply the loaded scaler (do not refit, use the same scaler fitted during training)
    X_scaled = scaler.transform(X_input)
    
    # Get predictions from both models
    pd_logit = logit_model.predict_proba(X_scaled)[:, 1][0]
    pd_gbm = gbm_model.predict_proba(X_scaled)[:, 1][0]
    
    # Define grades based on PD
    def grade_from_pd(pd_val):
        if pd_val < 0.02:
            return "A"
        elif pd_val < 0.05:
            return "B"
        elif pd_val < 0.10:
            return "C"
        elif pd_val < 0.20:
            return "D"
        else:
            return "E"
    
    grade_logit = grade_from_pd(pd_logit)
    grade_gbm = grade_from_pd(pd_gbm)
    
    return {
        "logit_score": pd_logit,
        "logit_grade": grade_logit,
        "gbm_score": pd_gbm,
        "gbm_grade": grade_gbm
    }

# Define the /explain endpoint to return SHAP explanations for the prediction
@app.post("/explain")
def explain(data: InputData):
    # Prepare the input data as an array (same as above)
    X_input = np.array([[data.avg_inflow, data.avg_outflow, data.avg_net_cf, data.std_net_cf, 
                         data.avg_balance, data.min_balance, data.max_drawdown, data.mom_net_cf_mean, 
                         data.mom_net_cf_std, data.mom_net_cf_p25, data.mom_net_cf_p75, 
                         data.avg_sales, data.avg_purchases, data.avg_va, data.tax_paid_mean, 
                         data.sales_std, data.purchases_std, data.sales_mom_mean, 
                         data.purchases_mom_mean]])

    # Apply the loaded scaler (do not refit, use the same scaler fitted during training)
    X_scaled = scaler.transform(X_input)
    
    # SHAP explanation for the LightGBM model
    explainer = shap.TreeExplainer(gbm_model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Extract SHAP values for LightGBM model
    feature_names = [
        "avg_inflow", "avg_outflow", "avg_net_cf", "std_net_cf", "avg_balance", "min_balance", 
        "max_drawdown", "mom_net_cf_mean", "mom_net_cf_std", "mom_net_cf_p25", "mom_net_cf_p75", 
        "avg_sales", "avg_purchases", "avg_va", "tax_paid_mean", "sales_std", "purchases_std", 
        "sales_mom_mean", "purchases_mom_mean"
    ]
    
    shap_values_dict = dict(zip(feature_names, shap_values[0]))
    
    return shap_values_dict
