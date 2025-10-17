from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app=FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting weather you have a heart disese or not using Machine Learning trained model",
    version="1.0.0"
)

model=joblib.load(r"D:\classification_proj\model_catboost_heart_disease.joblib")

class InputData(BaseModel):
    age: int
    sex: str                
    cp: str                 
    trestbps: int
    chol: int
    fbs: str                
    restecg: str            
    thalach: int
    exang: str              
    oldpeak: float
    slope: str              
    ca: int                 
    thal: str     

@app.get("/")
def home():
    return{"heart disease prediction app is running"}

@app.post("/Predict")
def predict(data:InputData):
    sex_map={"male":1,"female":0}
    cp_map = {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal pain": 2,
        "asymptomatic": 3
    }
    fbs_map = {"true": 1, "false": 0, "yes": 1, "no": 0}
    restecg_map = {
        "normal": 0,
        "st-t wave abnormality": 1,
        "left ventricular hypertrophy": 2
    }
    exang_map = {"yes": 1, "no": 0, "true": 1, "false": 0}
    slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
    thal_map = {
        "normal": 0,
        "fixed defect": 1,
        "reversible defect": 2
    }

    sex=sex_map.get(data.sex.lower(),0)
    cp = cp_map.get(data.cp.lower(), 0)
    fbs = fbs_map.get(data.fbs.lower(), 0)
    restecg = restecg_map.get(data.restecg.lower(), 0)
    exang = exang_map.get(data.exang.lower(), 0)
    slope = slope_map.get(data.slope.lower(), 1)
    thal = thal_map.get(data.thal.lower(), 1)


    X_input=np.array([[
    data.age, sex, cp, data.trestbps, data.chol, fbs,
    restecg, data.thalach, exang, data.oldpeak, slope, data.ca, thal]],dtype=float)
    
    Prediction=int(model.predict(X_input)[0])

    result = "Heart Disease Detected" if Prediction == 1 else "No Heart Disease"
    return {"prediction": result}