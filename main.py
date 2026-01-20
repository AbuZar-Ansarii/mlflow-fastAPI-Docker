import mlflow
import mlflow.sklearn
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.tracking import MlflowClient

iris = load_iris()
x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


mlflow.set_experiment("Iris_Classification2_Experiment")

with mlflow.start_run():
    # hyperparameters
    n_estimators = 50
    max_depth = 5

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("dataset", "Iris")
    mlflow.set_tag("run_type", "hyperparameter_tuning")

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="iris_RF_model",
        registered_model_name="Iris_Random_Forest_Model"
    )
    


    print(f"Model accuracy: {accuracy}")
    print(f"Model precision: {precision}")
    print(f"Model recall: {recall}")



