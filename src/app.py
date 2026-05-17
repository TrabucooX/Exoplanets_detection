import mlflow
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager


async def lifespan(app: FastAPI):
    # Start logic
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        model_uri = "runs:/39ba954e4c02466fa7624b250c55ce6e/Exoplanet_RandomForest_1"
        
        app.state.model = mlflow.sklearn.load_model(model_uri=model_uri)
        app.state.threshold = 0.65
        print("Production model loaded into the memory. Ready to perform inference.")
    except Exception as e:
        print(f"Error on startup loading assets: {str(e)}")

        app.state.model = None
        app.state.threshold = 0.65
    
    yield
    #Shutdown logic; preserving RAM
    print("Shutting down server, cleaning up assets.")
    if hasattr(app.state, "model"):
        del app.state.model


app = FastAPI(title="Exoplanet detection API", lifespan=lifespan)

class LightCurve(BaseModel):
    flux_values: list[float] = Field(..., min_length=100, max_length=100)
    kic_id: str = None

    class Config:
        json_schema_extra = {"example":{"kic_id": "KIC 7648362",
                                        "flux_values": [
                    round(float(1.0 - (0.02 * np.sin(x / 10)**2)), 5) 
                    for x in range(100)
                ]}}

@app.get("/")
def home(request: Request):
    return {"message": "Welcome to the Exoplanet Detection API",
            "status": "online",
            "model_available": getattr(request.app.state, "model") is not None,
            "inference_threshold": request.app.state.threshold}

@app.post("/predict")
def predict(lightcurve: LightCurve, request: Request):

    model = getattr(request.app.state, "model", None)
    threshold = request.app.state.threshold

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available currently.")
    
    # Load and prepare data
    try:
        flux_data = np.array(lightcurve.flux_values)
        df = pd.DataFrame([flux_data])
        df.columns = [f'flux_{i}' for i in range(100)]

        df["skew"] = df.iloc[:, :99].skew(axis=1).fillna(1.0)
        df["kurtosis"]  = df.iloc[:, :99].kurt(axis=1).fillna(1.0)
        df["mins"] = df.iloc[:, :99].min(axis=1)
        df["std"] = df.iloc[:, :99].std(axis=1)
        df["transit_depth"] = 1-df['mins']
        df['noise_to_signal'] = np.round(df['std']/df["transit_depth"], 6)
        df['kurtosis_to_depth'] = np.round(df['kurtosis']/df["transit_depth"], 6)
        features_df = df.drop(["mins", "transit_depth"], axis=1)

        print("Data preprocessed correctly. Preparing inference...")

        probabilities = model.predict_proba(features_df)[0, 1]
        is_exoplanet = bool(probabilities > threshold)

        return {"kic_id": lightcurve.kic_id,
                "is_candidate_for_exoplanet": is_exoplanet,
                "probability_candidate": np.round(probabilities, 3)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline failure: {str(e)}")
    