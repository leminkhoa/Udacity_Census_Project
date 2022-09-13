from pydantic import BaseModel, Field
from typing import Union
from enum import Enum

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
