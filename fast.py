import mlflow.sklearn
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
import logging



# load the model
model_name = 'Iris_Random_Forest_Model'

model_uri = f'models:/{model_name}/Production'

model = mlflow.sklearn.load_model(
    model_uri=model_uri
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ************* FAST API CODE ******************


# # fast api input data model
class IrisInput(BaseModel):
    data: List[List[float]]

# fast api app
app = FastAPI(title="Iris Prediction API",
    description="API for predicting iris flower species")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# helthcheck endpoint
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return {"status": "Model is loaded and API is healthy",
            "model_name": model_name,
            "service": 'Iris Prediction Service'
            }


# fast api endpoints
@app.get("/")
def read_root():
    return {"everything": "is Good",
            "docs": "/docs",
            "health": "/health"}

# prediction endpoint
@app.post('/prediction')
async def predict(input: IrisInput):
    logger.info(f"Received input data: {input.data}")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    try:
        if not input.data:
            raise HTTPException(status_code=400, detail="Input data is empty")
        for record in input.data:
            if len(record) != 4:
                raise HTTPException(status_code=400, detail="Each input must have exactly 4 features")
            
        data_np = np.array(input.data)  
        predictions = model.predict(data_np)
        return {
            'input': input.data,
            "predictions": predictions.tolist()
        }
    except ValueError as e:
        logger.error(f"ValueError in prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")