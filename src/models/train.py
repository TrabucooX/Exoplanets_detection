import pandas as pd
import matplotlib.pyplot as plt
from registry import MODELS
from evaluate import compute_metrics, plot_metrics
import mlflow
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import argparse
import warnings
warnings.filterwarnings('ignore')


def objective(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for multi-model tuning."""
    classifier_name = trial.suggest_categorical("classifier", ["xgboost", "lightgbm"])

    if classifier_name == "xgboost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.35, 0.45),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params, random_state=73)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.35, 0.45),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5)
        }
        model = LGBMClassifier(**params, verbose=-1, random_state=73)

    with mlflow.start_run(nested=True):
        model.fit(X_train, y_train)
        # We optimize for Macro F1 to ensure both Noise and Planet recall are balanced
        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average='macro')
        
        mlflow.log_params(params)
        mlflow.log_metric("macro_f1", score)
        return score

def train(model, model_version, tune=False):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Exoplanet Detection Pipeline")

    df = pd.read_parquet("data/exoplanets_flux_data.parquet")
    X = df.drop(["label", "transit_depth", "mins"], axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)
    
    dataset = mlflow.data.from_pandas(df, name="Full Dataset (2857 samples)", targets="label")

    with mlflow.start_run(run_name=f"{model}_{model_version}"):
        mlflow.log_input(dataset, context="training")

        if tune:
            print(f"Starting Hyperparameter Tuning for {model}...")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=25)
            
            mlflow.log_params(study.best_params)
            mlflow.log_metric("macro_f1", study.best_value)
            print(f"Best Trial Score: {study.best_value}")
            # Use the best parameters to train the final model
            best_params = study.best_params
            best_clf_name = best_params.pop('classifier')
            final_model = XGBClassifier(**best_params) if best_clf_name == 'xgboost' else LGBMClassifier(**best_params)
        else:
            final_model = MODELS[model]
        
        model_name = type(final_model).__name__
        final_model.fit(X_train, y_train)
        
        predictions = final_model.predict(X_test)
        prob_predictions = final_model.predict_proba(X_test)[:, 1]

        threshold = 0.65
        custom_preds = (prob_predictions > threshold).astype(int)
        
        metrics = compute_metrics(y_test, custom_preds)
        mlflow.log_metrics(metrics)

        plot_metrics(y_test, custom_preds, prob_predictions, model_name=model_name)
        
        mlflow.sklearn.log_model(sk_model=final_model, name=f"{model_name}_{model_version}")
        mlflow.set_tag("Training experiment", f"Basic {model_name} model for batch 1")
        




        
if __name__ == "__main__":
    print("Initializing model training...")
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("model_version", type=str, help="Model version")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning")

    args = parser.parse_args()

    if args.model not in MODELS.keys():
        print(f"Error: {args.model} is not a valid model. Choose from: {list(MODELS.keys())}")
        exit

    train(args.model, args.model_version, tune=args.tune)
    print("Run completed. Check MLflow UI for results.")
    

