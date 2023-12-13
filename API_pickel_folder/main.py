
"""machine learning API"""
import joblib
import numpy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

from typing import Union

# Define a Pydantic model for the input data
class InputData(BaseModel):
    features: list

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('model.pkl')

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input data to a format suitable for prediction
        prediction = model.predict([input_data.features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_random_model/")
async def train_random_model(sample_number: int, slope: float):
    """Train a random model"""
    try:
        x_train = numpy.random.rand(sample_number, 1)
        y_train = 1 + slope * x_train + numpy.random.rand(sample_number, 1)
        model = LinearRegression()
        model.fit(x_train, y_train)
        joblib.dump(model, 'dummy_model.pkl')
        return {"message": "model created successfully"}
    except HTTPException:
        return {"message": "model creation failed"}

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}