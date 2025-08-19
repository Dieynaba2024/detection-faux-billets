from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
from io import StringIO
import logging
import traceback
from pydantic import BaseModel
from typing import List
import os

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

app = FastAPI(debug=False, title="Fake Bills Detection API", version="1.0.0")

# Configuration CORS pour permettre les requêtes depuis Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour le développement, restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Chemin absolu pour Render
    model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.sav')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.sav')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Modèle et scaler chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle/scaler: {str(e)}")
    logger.error(traceback.format_exc())
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
        except UnicodeDecodeError:
            data = StringIO(contents.decode('cp1252'))
            df = pd.read_csv(data, sep=";")
        except Exception as e:
            logger.error(f"Erreur de lecture du fichier: {str(e)}")
            raise HTTPException(status_code=400, detail="Format de fichier invalide")
        
        logger.info(f"Colonnes reçues: {df.columns.tolist()}")
        logger.info(f"Nombre de lignes: {len(df)}")

        # Vérification des colonnes
        required_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Colonnes manquantes: {missing_cols}")
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes requises manquantes: {missing_cols}"
            )

        # Vérification des données manquantes
        if df[required_columns].isnull().any().any():
            logger.error("Données manquantes détectées")
            raise HTTPException(
                status_code=400,
                detail="Le fichier contient des données manquantes"
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
                "prediction": "Genuine" if pred == 1 else "Fake",
                "probability": float(prob[1] if pred == 1 else prob[0])
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
                "genuine_percentage": float(round(genuine_count / len(predictions) * 100, 2)) if len(predictions) > 0 else 0.0,
                "fake_percentage": float(round(fake_count / len(predictions) * 100, 2)) if len(predictions) > 0 else 0.0
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
    return {
        "message": "API de détection de faux billets",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Analyser un fichier CSV de billets"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Pour le développement local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apiEX:app", host="0.0.0.0", port=8000)
