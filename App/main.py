import uvicorn 
import joblib
import numpy as np
from pydantic import BaseModel
from datetime import date
from fastapi import FastAPI
import pickle


app = FastAPI()

class model_imput(BaseModel):
    dt : date

class OutputData(BaseModel):
    temp: float



# Loading the saved model
# model = joblib.load("linear_regression_model.pkl")

with open("linear_regression.pkl", 'rb') as f:
    model = pickle.load(f)

print('model loaded')

@app.get('/')
def index():
    return "Hello World"


@app.get('/Welcome/{name}')
def get_name(name: str):
    return f'Welcome {name}'

@app.post('/predict')
def temp_pred(data : model_imput):
    year = data.dt.year
    print(year)
    X = np.array([year]).reshape(-1, 1)
    print(X)
    y = round(model.predict(X)[0], 3)
    print(y)
    return OutputData(temp=y)



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)




