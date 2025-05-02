from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from models.BaseModel import BaseModel

from environs import load_dotenv
import os

from exceptions.TrainingException import TrainingException
from exceptions.InternalException import InternalException

from selector import select_model

load_dotenv()
app = FastAPI()

model = os.getenv("MODEL")
print(model)


model = select_model(model_name=model)
def get_model() -> BaseModel:
    return model

@app.post("/learn")
async def learn(model = Depends(get_model)):
    try:
        model.learn()
        return JSONResponse(status_code=200, content={"message": "Model trained successfully."})
    except InternalException as e:
        return JSONResponse(status_code=500, content={"message": "Internal server error. Please try again later."})
    except TrainingException as e:
        return JSONResponse(status_code=400, content={"message": "Error while training."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post('/test')
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"message": "File format must be CSV."})

    try:
        content = await file.read()
        return model.predict(content)
    except TrainingException as e:
        return JSONResponse(status_code=400, content={"message": "Missing data for testing."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Internal server error. Please try again later."})
