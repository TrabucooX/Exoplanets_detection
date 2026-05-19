import pandas as pd
import numpy as np
import mlflow
import argparse
import os
import joblib


def predict(data_path, model_name, model_path="models/production/ensemble_model.pkl", threshold=0.5):
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Exoplanet Inference Production")

    print(f"Loading unlabeled data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data has not been found at {data_path}")
    
    unlabeled_exoplanets = pd.read_parquet(data_path)
    X_unlabeled = unlabeled_exoplanets.drop(["kic_id", "transit_depth", "mins"], axis=1)

    dataset_metadata = mlflow.data.from_pandas(unlabeled_exoplanets, name=os.path.basename(data_path))

    # Start an MLflow Production Inference tracking run
    with mlflow.start_run(run_name=f"Inference_Batch_{os.path.basename(data_path)}"):
        # Log dataset lineage
        mlflow.log_input(dataset_metadata, context="inference")

        print(f"Fetching production ensemble from: {model_path}...")
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Please check your path.")
        model = joblib.load(model_path)

        mlflow.set_tag("deployed_model", model_name)
        mlflow.log_param("applied_threshold", threshold)
        mlflow.sklearn.log_model(sk_model=model, name=model_name)

        probabilities = model.predict_proba(X_unlabeled)[:, 1]
        predictions = (probabilities > threshold).astype(int)
        
        # Build candidate framework dataframe
        output_df = pd.DataFrame({
            "prediction_label": predictions,
            "exoplanet_probability": np.round(probabilities, 4)
        })
        if "kic_id" in unlabeled_exoplanets.columns:
            output_df.insert(0, "kic_id", unlabeled_exoplanets["kic_id"])
            
        # Log high-level inference performance metadata metrics
        planet_count = int(np.sum(predictions))
        mlflow.log_metric("total_items_screened", len(output_df))
        mlflow.log_metric("candidates_discovered", planet_count)
        mlflow.log_metric("flagged_noise_count", len(output_df) - planet_count)
        
        # Export data locally
        output_filename = "exoplanets_predictions_output.csv"
        output_df.to_csv(output_filename, index=False)
        
        # Log the output predictions directly to MLflow storage as an artifact
        mlflow.log_artifact(output_filename)
        print(f"Production inference logged successfully to MLflow. Discoveries: {planet_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Exoplanet Vetting Pipeline - Inference")
    parser.add_argument("--data", type=str, required=True, help="Path to the unlabeled Parquet data")
    parser.add_argument("--model_path", type=str, default="models/production/ensemble_model.pkl", help="The path containing the production or desired model")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the desired model in MLflow")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold boundary")
    
    args = parser.parse_args()
    predict(data_path=args.data, model_path=args.model_path, model_name=args.model_name, threshold=args.threshold)


