import mlflow


#  REGISTER THE MODEL

# model_name = 'iris_knight_model'
# run_id = 'a3cd3f5c4212483eb2f624ad0de1c101'

# model_uri = f'runs:/{run_id}/{model_name}'  


# result = mlflow.register_model(
#     model_uri,
#     model_name
# )


#  LOAD THE MODEL


model_name = 'Iris_Random_Forest_Model'
alias_name = 'champion'  

# Use the alias name directly
model_uri = f'models:/{model_name}@{alias_name}'

model = mlflow.sklearn.load_model(
    model_uri=model_uri
)

print("Model loaded successfully.")


user_input = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(user_input)
print(f"Prediction for input {user_input}: {prediction}")
