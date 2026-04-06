import os
import sys
import requests
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core components for model deserialization
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.cols_to_keep_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.cols_to_keep_ = [i for i in range(df.shape[1]) if i not in to_drop]
        return self

    def transform(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        if self.cols_to_keep_ is None:
            return X_val
        return X_val[:, self.cols_to_keep_]

# 🛠️ CRITICAL: Inject into main module for joblib deserialization
sys.modules['__main__'].CorrelationFilter = CorrelationFilter

app = FastAPI()

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "horse_colic_model_svm.joblib")

templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")

# Load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded from: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")

# GROQ Configuration (Set via Vercel Environment Variables)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class ClinicalData(BaseModel):
    pulso: float
    temp_rectal: float
    dolor: int
    volumen_celular: float
    proteina_total: float
    edad: int
    lesion_quirurgica: int

# Full list of 46 columns expected by the model pipeline
MODEL_COLUMNS = [
    "temp_rectal", "pulso", "frec_respiratoria", "ph_reflujo", "volumen_celular", 
    "proteina_total", "proteina_abdominocentesis", "cirugia_2.0", "edad_9", 
    "temp_extremidades_2.0", "temp_extremidades_3.0", "temp_extremidades_4.0", 
    "pulso_periferico_2.0", "pulso_periferico_3.0", "pulso_periferico_4.0", 
    "membranas_mucosas_2.0", "membranas_mucosas_3.0", "membranas_mucosas_4.0", 
    "membranas_mucosas_5.0", "membranas_mucosas_6.0", "tiempo_llenado_capilar_2.0", 
    "tiempo_llenado_capilar_3.0", "dolor_2.0", "dolor_3.0", "dolor_4.0", "dolor_5.0", 
    "peristalsis_2.0", "peristalsis_3.0", "peristalsis_4.0", "distension_abdominal_2.0", 
    "distension_abdominal_3.0", "distension_abdominal_4.0", "sonda_nasogastrica_2.0", 
    "sonda_nasogastrica_3.0", "reflujo_nasogastrico_2.0", "reflujo_nasogastrico_3.0", 
    "heces_rectal_2.0", "heces_rectal_3.0", "heces_rectal_4.0", "abdomen_2.0", 
    "abdomen_3.0", "abdomen_4.0", "abdomen_5.0", "apariencia_abdominocentesis_2.0", 
    "apariencia_abdominocentesis_3.0", "lesion_quirurgica_2"
]

# ⚖️ Scaling statistics extracted from Activity 1 (CRITICAL for model accuracy)
SCALING_STATS = {
    "temp_rectal": {"mean": 38.0605, "std": 0.617},
    "pulso": {"mean": 68.7725, "std": 27.3984},
    "frec_respiratoria": {"mean": 26.5702, "std": 14.2952},
    "ph_reflujo": {"mean": 4.9313, "std": 0.7773},
    "volumen_celular": {"mean": 46.6433, "std": 9.2356},
    "proteina_total": {"mean": 40.5421, "std": 30.2376},
    "proteina_abdominocentesis": {"mean": 1.5868, "std": 1.3976}
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(data: ClinicalData):
    try:
        logger.info(f"Incoming prediction request: {data}")
        # Construct 46 columns vector (RAW)
        input_vector = {col: 0.0 for col in MODEL_COLUMNS}
        
        # 1. Numerical mappings (RAW values)
        input_vector['pulso'] = data.pulso
        input_vector['temp_rectal'] = data.temp_rectal
        input_vector['volumen_celular'] = data.volumen_celular
        input_vector['proteina_total'] = data.proteina_total
        # Defaults for missing numericals (using reasonable clinical values)
        input_vector['frec_respiratoria'] = 24.0
        input_vector['ph_reflujo'] = 4.5
        input_vector['proteina_abdominocentesis'] = 2.0
        
        # 2. Categorical mappings (Manual One-Hot)
        if data.lesion_quirurgica == 1: input_vector['lesion_quirurgica_2'] = 0.0 # 1=Yes, 2=No
        else: input_vector['lesion_quirurgica_2'] = 1.0
        
        if data.edad == 2: input_vector['edad_9'] = 1.0 
        
        # Pain mapping
        if data.dolor == 2: input_vector['dolor_2.0'] = 1.0
        elif data.dolor == 3: input_vector['dolor_3.0'] = 1.0
        elif data.dolor == 4: input_vector['dolor_4.0'] = 1.0
        elif data.dolor == 5: input_vector['dolor_5.0'] = 1.0
        
        # 3. 🧪 CRITICAL: Apply Scaling (Standardization)
        # The model expects (X - mean) / std as provided in SCALING_STATS
        for col, stats in SCALING_STATS.items():
            if col in input_vector:
                input_vector[col] = (input_vector[col] - stats['mean']) / stats['std']
        
        # Construct DataFrame in the EXACT expected column order
        df_input = pd.DataFrame([input_vector], columns=MODEL_COLUMNS)
        logger.info(f"Standardized Input (Head): {input_vector['pulso']}, {input_vector['temp_rectal']}")
        
        # Outcome map (Matching the unifed notebook: 0: Lived, 1: Died, 2: Euthanized)
        outcome_map = {0: "LIVED (SOBREVIVE)", 1: "DIED (MUERE)", 2: "EUTHANIZED (EUTANASIA)"}
        
        if model:
            # Predict
            pred_idx = model.predict(df_input)[0]
            prediction = outcome_map.get(pred_idx, f"Resultado Desconocido ({pred_idx})")
            logger.info(f"Prediction result: {prediction}")
        else:
            prediction = "Error: Modelo no cargado"
            logger.warning("Model not found during prediction")

        # 3. Generate Report
        report = generate_ai_report(data, prediction)
        return {"prediction": prediction, "report": report}
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}", exc_info=True)
        return {"prediction": "Error", "report": f"Internal Error: {str(e)}"}

def generate_ai_report(data, result):
    if not GROQ_API_KEY or "gsk" not in GROQ_API_KEY:
        return f"Informe del Sistema: El paciente tiene un resultado predicho de {result}. Configure su API Key para un análisis clínico completo."

    prompt = f"""Analiza este caso veterinario de cólico equino:
    - Pulso: {data.pulso} bpm
    - Temperatura: {data.temp_rectal} °C
    - Nivel de Dolor: {data.dolor}/5
    - Hematocrito (PCV): {data.volumen_celular}%
    - Proteína Total: {data.proteina_total} g/dL
    - Predicción de IA: {result}

    Actúa como un agente experto y genera un informe breve (máx 120 palabras) en español explicando por qué los signos clínicos llevan a este riesgo. Usa un tono profesional."""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"Groq API Error ({response.status_code}): {response.text}")
            return f"Error API ({response.status_code}): {response.text}"
    except Exception as e:
        logger.error(f"Error calling Groq: {str(e)}")
        return f"Error al generar informe: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
