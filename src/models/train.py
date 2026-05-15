import pandas as pd
import matplotlib.pyplot as plt
from registry import MODELS
from evaluate import compute_metrics, plot_metrics
import mlflow
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')


def train(model, model_version):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Exoplanet Detection Pipeline")

    df = pd.read_parquet("data/exoplanets_flux_data.parquet")
    X = df.drop(["label", "transit_depth", "mins"], axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)
    
    dataset = mlflow.data.from_pandas(df, name="Full Dataset (2857 samples)", targets="label")


    with mlflow.start_run(run_name=f"{model}_{model_version}"):
        mlflow.log_input(dataset, context="training")

        model = MODELS[model]
        model_name = type(model).__name__
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        prob_predictions = model.predict_proba(X_test)[:, 1]

        threshold = 0.65
        custom_preds = (prob_predictions > threshold).astype(int)
        
        metrics = compute_metrics(y_test, predictions)
        mlflow.log_metrics(metrics)

        plot_metrics(y_test, predictions, prob_predictions, model_name=model_name)
        
        mlflow.sklearn.log_model(sk_model=model, name=f"{model_name}_{model_version}")
        mlflow.set_tag("Training experiment", f"Basic {model_name} model for batch 1")
        




        
if __name__ == "__main__":
    print("Initializing model training...")
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("model_version", type=str, help="Model version")


    args = parser.parse_args()

    if args.model not in MODELS.keys():
        print(f"Error: {args.model} is not a valid model. Choose from: {list(MODELS.keys())}")
        exit

    train(args.model, args.model_version)
    print("Run completed. Check MLflow UI for results.")
    

