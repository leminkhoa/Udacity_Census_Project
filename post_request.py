import requests
import json

infer_endpoint = 'https://mlmodel-fastapi-khoale.herokuapp.com/inferences/'

body = {
    "age": 25,
    "workclass": "Federal-gov",
    "fnlgt": 35,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Divorced",
    "occupation": "Farming-fishing",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 15,
    "native_country": "United-States"
}

data = json.dumps(body)
r = requests.post(infer_endpoint, data=data)

print("Status code:", r.status_code)
print("Response body:", r.json()['pred_salary'])
