from python:3.10-slim

LABEL maintainer="Hania Nasir"
LABEL project="Heart Disease Prediction"
LABEL description="A Dockerized machine learning app for predicting weather a person has heart disease or not using catboost and streamlit"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]