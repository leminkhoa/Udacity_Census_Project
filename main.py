import pandas as pd
import joblib
import logging
from hydra import compose, initialize
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Union
from enum import Enum
from starter.ml.data import process_data
from starter.ml.model import inference

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Initialize app
app = FastAPI()

# Import config
with initialize(version_base=None, config_path="starter/experiments"):
    config = compose(config_name="ml_config")


class SalaryResponse(str, Enum):
    lte_50k = '<=50K'
    gt_50k = '>50K'


class Response(BaseModel):
    pred_salary: SalaryResponse


# Define input schema
class Item(BaseModel):
    # insert type hinting here for feature inputs
    age: int = Field(example=25)
    workclass: str = Field(default=None, example="Federal-gov",)
    fnlgt: int = Field(default=None, example=35,)
    education: str = Field(default=None, example="Bachelors",)
    education_num: int = Field(default=None, example=13)
    marital_status: str = Field(default=None, example='Divorced')
    occupation: str = Field(default=None, example='Farming-fishing')
    relationship: str = Field(default=None, example='Husband')
    race: str = Field(default=None, example='White')
    sex: str = Field(default=None, example='Male')
    capital_gain: Union[int, float] = Field(default=None, example=0.0)
    capital_loss: Union[int, float] = Field(default=None, example=0.0)
    hours_per_week: int = Field(default=None, example=15)
    native_country: str = Field(default=None, example='United-States')


# GET method.
@app.get("/")
async def greetings():
    return {"message": "Hello World!"}


# POST method.
@app.post("/inferences/", responses={200: {"model": Response}})
async def predict(item: Item):
    try:
        # load user inputs
        input_json = {k: [v] for k, v in dict(item).items()}
        logger.info(f"Load input \n {input_json}")
        input_df = pd.DataFrame.from_dict(input_json)
        logger.info("Converted input to dataframe")

        # load pipeline components
        encoder = joblib.load(config.modeling.encoder_output_path)
        lb = joblib.load(config.modeling.lb_output_path)
        cat_features = config.main.categorical_cols
        logger.info("Loaded pipeline components")

        # load trained model
        clf = joblib.load(config.modeling.model_output_path)
        logger.info("Loaded model")

        # Process data
        logger.info("Processing input")
        X, _, _, _ = process_data(
            input_df, categorical_features=cat_features, label=None, encoder=encoder, lb=lb, training=False
        )

        # predict
        preds = inference(clf, X)
        # Convert back
        salary = lb.inverse_transform(preds)[0]
    except Exception:
        raise HTTPException(status_code=500, detail='Internal Server Error')
    else:
        return {"pred_salary": salary}
