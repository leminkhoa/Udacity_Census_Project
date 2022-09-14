import json
import requests

root_endpoint = 'https://mlmodel-fastapi-khoale.herokuapp.com/'
infer_endpoint = 'https://mlmodel-fastapi-khoale.herokuapp.com/inference/'


def test__get_return_200():
    '''Test remote GET method, should return 200 and Hello World message'''
    r = requests.get(root_endpoint)
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World!"}


def test__post_return_200():
    '''Test remote POST method, should return 200 and predicted salary'''
    input = {
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
    data = json.dumps(input)
    r = requests.post(infer_endpoint, data=data)
    assert r.status_code == 200
    assert r.json()['pred_salary'] in ['<=50K', '>50K']


def test__post_empty_input_return_422():
    '''Test remote POST method from empty body, should return 422'''
    input = dict()
    data = json.dumps(input)
    r = requests.post(infer_endpoint, data=data)
    assert r.status_code == 422


def test__post_missing_fields_return_500():
    '''Test remote POST method with body having wrong schema, should return 500'''
    input = {
        "age": 25,
    }
    data = json.dumps(input)
    r = requests.post(infer_endpoint, data=data)
    assert r.status_code == 500
