# Heart-Disease-Prediction
This project predicts heart disease in patients based on clinical and demographic attributes such as age, gender, cholesterol level, blood pressure, and chest pain type.
The main goal is to assist healthcare professionals in early risk detection using machine learning classification techniques.

## Project Overview
Heart disease remains one of the major causes of death globally.
This project applies supervised machine learning models to predict whether a patient is likely to have heart disease based on their medical data.
It demonstrates how data-driven models can support healthcare decision-making and preventive diagnosis

## Tools and Technologies
Programming Language: Python
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM, CatBoost
Model Training: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
Model Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
Deployment Frameworks: Streamlit (UI) and FastAPI (backend API)
Other Tools: Git, GitHub, VS Code

## Project Structure
heart-disease-prediction/
│
├── .gitignore
├── app.py                        
├── api_fastapi.py              
├── Heart_disease_classification.ipynb 
├── model_catboost_heart_disease.joblib 
├── requirements.txt              
├── Dockerfile                   
└── README.md      

## Machine Learning Workflow

### Data Preprocessing
-Loaded and explored the heart disease dataset
-Encoded categorical features such as Sex, ChestPainType, ExerciseAngina, and ST_Slope
-Scaled numerical features using StandardScaler
-Split the dataset into training and testing sets

### Model Training
-Trained multiple classification models including:
-Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost
-Used GridSearchCV for hyperparameter tuning
-Evaluated models using Accuracy, Precision, Recall, F1-score, and ROC-AUC
-Saved the best model as model_catboost_heart_disease.joblib using joblib

### Author
- Hania Nasir

### Deployment
- Created a Streamlit app for interactive user predictions  
- Saved trained models as `.joblib` files and connected them to the app  

---
