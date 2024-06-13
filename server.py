from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load the trained Linear Regression model
lr_model = joblib.load("lr_model.pkl")

# Define the input data schema
class InputData(BaseModel):
    cpu_usage: float
    memory_usage: float
    network_traffic: float
    power_consumption: float
    num_executed_instructions: float
    execution_time: float

app = FastAPI()

# Define the prediction endpoint
data = pd.read_csv("vmCloud_data.csv")

# Define an endpoint to accept data from a CSV file
@app.post("/predict_from_csv/")
async def predict_energy_efficiency_from_csv():
    try:
        # Prepare the input data for prediction
        input_data = np.array([[data.cpu_usage, data.memory_usage, data.network_traffic, 
                                 data.power_consumption, data.num_executed_instructions, 
                                 data.execution_time]])
        
        # Make predictions
        prediction = lr_model.predict(input_data)
        
        # Return the prediction
        return {"energy_efficiency_prediction": prediction[0]}
    
    except Exception as e:
        # In case of any errors, return an HTTP Exception
        raise HTTPException(status_code=500, detail=str(e))
