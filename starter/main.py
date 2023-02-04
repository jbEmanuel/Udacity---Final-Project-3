# Put the code for your API here.
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import numpy as np
import pickle

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

# Load the model 

pickled_model_lrc = pickle.load(open('./model/rfc_model.pkl', 'rb'))


# Define a GET on the specified endpoint.
@app.get("/")
async def describe_api():
    return {"description": "This api must return the salary category of a person, wheter if its above 50k or bellow"}

# This allows sending of data (our Features) via POST to the API.
@app.post("/predict/")
async def predict_salary(features: Features):
    features = features.dict()
    fnlgt = features['fnlgt']
    age = features['age']
    capital_gain = features['capital_gain']
    hours_per_week = features['hours_per_week']
    marital_status_Married_civ_spouse= features['civilian_spouse']
    education_num= features['education_num']
    relationship_Husband= features['relationship_Husband']
    capital_loss= features['capital_loss']

    if marital_status_Married_civ_spouse == 'true':
        marital_status_Married_civ_spouse = 1
    else:
        marital_status_Married_civ_spouse = 0
    
    if relationship_Husband == 'true':
        relationship_Husband = 1
    else:
        relationship_Husband = 0


    prediction = pickled_model_lrc.predict(np.array([[fnlgt,age,capital_gain,hours_per_week,marital_status_Married_civ_spouse, education_num,relationship_Husband, capital_loss]]))
    
    if(prediction[0]> 0):
        prediction="bellow 50k"
    else:
        prediction="above 50k"
    
    
    return {
        "Salary": prediction
         }