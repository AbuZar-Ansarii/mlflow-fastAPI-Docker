import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
import numpy as np

# from typing import List

# app = FastAPI()


# model_name = 'Iris_Random_Forest_Model'
# model_version = 6

# model_uri = f'models:/{model_name}/{model_version}'

# model = mlflow.sklearn.load_model(
#     model_uri=model_uri
# )

# # fast api endpoints
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.post('/predict')
# def predict(data: List[List[float]]):
#     data_np = np.array(data)
#     predictions = model.predict(data_np)
#     return {'input': data,
#         "predictions": predictions.tolist()}


from mlflow.tracking import MlflowClient

# client = MlflowClient()

# # List all registered models with their stages
# registered_models = client.search_registered_models()

# for model in registered_models:
#     print(f"\nModel: {model.name}")
#     for version in model.latest_versions:
#         print(f"  Version {version.version}: Stage='{version.current_stage}', Aliases={version.aliases}")


# client = MlflowClient()

model_name = 'Iris_Random_Forest_Model'
# model_version = 7

# client.transition_model_version_stage(
#     name=model_name,
#     version=model_version,
#     stage='Production',
#     archive_existing_versions=False
# )
 

model_uri = f'models:/{model_name}/Production'

model = mlflow.sklearn.load_model(
    model_uri=model_uri
)

print(f"Model {model_name} loaded successfully.")