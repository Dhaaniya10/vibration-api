from fastapi import FastAPI
from pydantic import BaseModel
from autogluon.tabular import TabularPredictor
import pandas as pd

app = FastAPI()
predictor = TabularPredictor.load("BestOnlyModel")  # Use only best model folder

class InputData(BaseModel):
    data: dict

@app.post("/predict")
def predict(input_data: InputData):
    df = pd.DataFrame([input_data.data])
    prediction = predictor.predict(df)
    return {"prediction": str(prediction.iloc[0])}
