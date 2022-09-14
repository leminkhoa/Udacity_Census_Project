from fastapi import FastAPI
from fastapi import HTTPException
from training.ml.schema import Item, Response
from training.infer_model import predict_salary

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Initialize app
app = FastAPI()


# GET method.
@app.get("/")
async def greetings():
    return {"message": "Hello World!"}


# POST method.
@app.post("/inferences/", responses={200: {"model": Response}})
async def predict(item: Item):
    try:
        salary = predict_salary(item)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return {"pred_salary": salary}
