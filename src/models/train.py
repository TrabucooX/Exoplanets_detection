import pandas as pd
import matplotlib.pyplot as plt
from registry import MODELS
from evaluate import compute_metrics, plot_confusion_matrix_rocauc
import mlflow
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')


def train(model, model_version):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Models training")

    df = pd.read_parquet("data/exoplanets_interpolated_flux_data_batch_1.parquet")
    X_train, X_test, y_train, y_test = train_test_split(df.drop("label", axis=1), df["label"], random_state=73)
    
    dataset = mlflow.data.from_pandas(df, name="Batch 1", targets="label")


    with mlflow.start_run(run_name="Model alpha"):
        mlflow.log_input(dataset, context="training")

        model = MODELS[model]
        model_name = type(model).__name__
        model.fit(X_train, y_train)
        model_info = mlflow.sklearn.log_model(sk_model=model, name=f"{model_name}_{model_version}")
        
        predictions = model.predict(X_test)
        prob_predictions = model.predict_proba(X_test)[:, 1]
        
        metrics = compute_metrics(y_test, predictions)
        plot_confusion_matrix_rocauc(y_test, predictions, prob_predictions, model_name=model_name)
        mlflow.log_metrics(metrics)

        mlflow.set_tag("Training experiment", f"Basic {model_name} model for batch 1")
        




        
if __name__ == "__main__":
    print("Model trained and successfully tracked by mlflow.\nHere are the results:")
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("model_version", type=str, help="Model version")


    args = parser.parse_args()

    if args.model not in MODELS.keys():
        print(f"Error: {args.model} is not a valid model. Choose from: {list(MODELS.keys())}")
        exit

    train(args.model, args.model_version)
    

