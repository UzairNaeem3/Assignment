import uvicorn 
import joblib
import numpy as np
from pydantic import BaseModel
from datetime import date
from fastapi import FastAPI


app = FastAPI()

class model_imput(BaseModel):
    dt : str


# Loading the saved model
model = joblib.load("linear_regression_model.pkl")


@app.get('/')
def index():
    return "Hello World"


@app.get('/Welcome/{name}')
def get_name(name: str):
    return f'Welcome {name}'

@app.post('/predict')
def temp_pred(data : model_imput):
    parts = data.dt.split("-")
    year = int(parts[0])
    print(year)
    X = np.array([year]).reshape(-1,1)
    print(model.predict(X)[0])



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)




