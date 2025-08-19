from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
from io import StringIO
import logging
import traceback
import uvicorn
from pydantic import BaseModel
from typing import List


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles pour la réponse
class PredictionResult(BaseModel):
    id: int
    prediction: str
    probability: float

class StatsResult(BaseModel):
    total: int
    genuine: int
    fake: int
    genuine_percentage: float
    fake_percentage: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    stats: StatsResult

app = FastAPI(debug=True)

# Fonction pour convertir les types NumPy en types Python natifs
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Chargement du modèle et du scaler
try:
    model = joblib.load('random_forest_model.sav')
    scaler = joblib.load('scaler.sav')
    logger.info("Modèle et scaler chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle/scaler: {str(e)}")
    raise RuntimeError("Impossible de charger le modèle ou le scaler") from e

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Lire le fichier CSV
        contents = await file.read()
        logger.info(f"Fichier reçu: {file.filename}, taille: {len(contents)} bytes")
        
        # Détection de l'encodage et séparateur
        try:
            data = StringIO(contents.decode('utf-8'))
            df = pd.read_csv(data, sep=";")
        except:
            data = StringIO(contents.decode('cp1252'))
            df = pd.read_csv(data, sep=";")
        
        logger.info(f"Colonnes reçues: {df.columns.tolist()}")
        logger.info(f"Exemple de données:\n{df.head().to_string()}")

        # Vérification des colonnes
        required_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Colonnes manquantes: {missing_cols}")
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes requises manquantes: {missing_cols}"
            )

        # Standardisation des données
        X = df[required_columns]
        X_scaled = scaler.transform(X)
        
        # Prédictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Conversion des types NumPy et formatage des résultats
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "id": int(i),
                "prediction": "Genuine" if pred else "Fake",
                "probability": float(prob[1] if pred else prob[0])
            })
        
        # Statistiques globales
        genuine_count = int(sum(predictions))
        fake_count = int(len(predictions) - genuine_count)
        
        response = {
            "predictions": results,
            "stats": {
                "total": int(len(predictions)),
                "genuine": genuine_count,
                "fake": fake_count,
                "genuine_percentage": float(round(genuine_count / len(predictions) * 100, 2)),
                "fake_percentage": float(round(fake_count / len(predictions) * 100, 2))
            }
        }
        
        # Conversion finale
        return convert_numpy_types(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du serveur"
        )

@app.get("/")
async def root():
    return {"message": "API de détection de faux billetsZ"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
