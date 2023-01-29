# Put the code for your API here.
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object with its components and their type.
class Features(BaseModel):
    fnlgt: int 
    age: int 
    capital_gain: int 
    hours_per_week: int 
    civilian_spouse: bool 
    education_num: int 
    relationship_Husband: bool
    capital_loss: int 


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

# This allows sending of data (our Features) via POST to the API.
@app.post("/predict/")
async def predict_salary(features: Features):
    return features