import pandas as pd
import matplotlib.pyplot as plt
from registry import MODELS
from evaluate import compute_metrics, plot_metrics
import mlflow
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import argparse
import warnings
warnings.filterwarnings('ignore')


def objective(trial, X_train, y_train, X_test, y_test, model_to_tune, dataset):
    """Optuna objective function for multi-model tuning."""

    if model_to_tune == "xgb":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 1.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params, random_state=73)
    elif model_to_tune == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 1.2),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5)
        }
        model = LGBMClassifier(**params, verbose=-1, random_state=73)

    elif model_to_tune == "rf":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        }
        model = RandomForestClassifier(**params, random_state=73)
    
    with mlflow.start_run(nested=True):
        model.fit(X_train, y_train)
       
        predictions = model.predict(X_test)
        score = f1_score(y_test, predictions, average='macro')
        
        mlflow.log_params(params)
        mlflow.log_metric("macro_f1", score, dataset=dataset)
        return score
    
def get_production_ensemble(model_version):
    """
    Dynamically loads the chosen registered models from MLflow
    that are to be used inside a StackingClassifier as base estimators.
    """
    print("Loading base estimators models from MLflow Models...")
    
    rf_tuned = mlflow.sklearn.load_model(f"runs:/2f9f9ccc025e410fa29a848fc0cac406/Exoplanet_RandomForest_{model_version}")
    xgb_tuned = mlflow.sklearn.load_model(f"runs:/37116ea5a09f45cf9a6b18bd2b5f03ac/Exoplanet_XGBoost_{model_version}")
    lgbm_tuned = mlflow.sklearn.load_model(f"runs:/aee16319ad7742fcb6d7b093d8a0229e/Exoplanet_LightGBM_{model_version}")
    
    base_estimators = [
        ('rf', rf_tuned),
        ('xgb', xgb_tuned),
        ('lgbm', lgbm_tuned)
    ]

    return StackingClassifier(estimators=base_estimators, cv=3)


def train(mode, model_version, model_to_tune=None):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Exoplanet Detection Pipeline")

    df = pd.read_parquet("data/exoplanets_flux_data.parquet")
    X = df.drop(["label", "transit_depth", "mins"], axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)
    
    dataset = mlflow.data.from_pandas(df, name="Full Dataset", targets="label")

    with mlflow.start_run(run_name=f"Training_{mode}_{model_version}"):
        mlflow.log_input(dataset, context=mode)
        mlflow.set_tag("pipeline_mode", mode)

        if mode == "tune":
            print(f"Starting Optuna Hyperparameter Tuning for {model_to_tune}...")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, model_to_tune, dataset), n_trials=20)
            
            best_params = study.best_params
            mlflow.log_params(best_params)
       
            if model_to_tune == "xgb":
                final_model = XGBClassifier(**best_params, random_state=73)
                registry_name = f"Exoplanet_XGBoost_{model_version}"
            elif model_to_tune == "lgbm":
                final_model = LGBMClassifier(**best_params, verbose=-1, random_state=73)
                registry_name = f"Exoplanet_LightGBM_{model_version}"
            elif model_to_tune == "rf":
                final_model = RandomForestClassifier(**best_params, random_state=73)
                registry_name = f"Exoplanet_RandomForest_{model_version}"

            print(f"Best Trial Score: {study.best_value}")

        elif mode == 'ensemble':
            print("Training Final Production Stacking Ensemble...")
            final_model = get_production_ensemble(model_version)
            registry_name = f"StackingClassifier_{model_version}"

            mlflow.log_params(final_model.get_params())

        final_model.fit(X_train, y_train)
        predictions = final_model.predict(X_test)

        score = f1_score(y_test, predictions, average='macro')

        prob_predictions = final_model.predict_proba(X_test)[:, 1]

        threshold = 0.6
        custom_predictions = (prob_predictions > threshold).astype(int)
        
        metrics = compute_metrics(y_test, predictions)
        plot_metrics(y_test, predictions, prob_predictions, model_name=registry_name)
        
        model_info = mlflow.sklearn.log_model(sk_model=final_model, name=registry_name)
        mlflow.log_metric("macro_f1", score, dataset=dataset, model_id=model_info.model_id)
        mlflow.log_metrics(metrics, dataset=dataset, model_id=model_info.model_id)
        
        




        
if __name__ == "__main__":
    print("Initializing model training...")
    parser = argparse.ArgumentParser(description="Unified Exoplanet Training & Tuning Pipeline")

    parser.add_argument("--mode", type=str, required=True, choices=["tune", "ensemble"])
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["xgb", "lgbm", "rf"], 
                        help="Required if --mode is 'tune'. Which model to optimize.")
    
    args = parser.parse_args()
    
    if args.mode == "tune" and not args.model:
        parser.error("--model is required when --mode is set to 'tune'!")

    args = parser.parse_args()

    train(args.mode, args.version, args.model)
    print("Run completed. Check MLflow UI for results.")
    

