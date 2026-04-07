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

# ── Deserialization & Pipeline Logic ───
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.cols_to_keep_ = None
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        return X_val[:, self.cols_to_keep_] if self.cols_to_keep_ is not None else X_val

sys.modules['__main__'].CorrelationFilter = CorrelationFilter

app = FastAPI()

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# 🏆 THE FINAL SPECIALIZED MODEL (V7 - 19 FEATURES ONLY)
MODEL_PATH = os.path.join(APP_DIR, "models", "horse_colic_model_final_v7.joblib")

templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")

# Load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ SPECIALIZED V7 loaded: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Error loading model V7: {e}")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # Debe configurarse en el Dashboard de Vercel
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class ClinicalData(BaseModel):
    pulso: float
    temp_rectal: float
    volumen_celular: float
    proteina_total: float
    dolor: int
    edad: int
    lesion_quirurgica: int
    heces: int
    mucosas: int
    # Form auxiliary (hidden for V7)
    surgery: int = 1

# EXACT 19 COLUMNS EXPECTED BY V7 SPECIALIZED MODEL
MODEL_COLUMNS = [
    'temp_rectal', 'pulso', 'volumen_celular', 'proteina_total',
    'cirugia_2.0', 'edad_9', 'lesion_quirurgica_2',
    'dolor_2.0', 'dolor_3.0', 'dolor_4.0', 'dolor_5.0',
    'heces_rectal_2.0', 'heces_rectal_3.0', 'heces_rectal_4.0',
    'membranas_mucosas_2.0', 'membranas_mucosas_3.0', 'membranas_mucosas_4.0', 'membranas_mucosas_5.0', 'membranas_mucosas_6.0'
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(data: ClinicalData):
    try:
        # Initialize 19 dimensions
        input_dict = {col: 0.0 for col in MODEL_COLUMNS}
        
        # 1. Z-Score Scaling (Using stats from the 19-feature training subset)
        input_dict['temp_rectal'] = (data.temp_rectal - 38.06) / 0.61
        input_dict['pulso'] = (data.pulso - 68.7) / 27.3
        input_dict['volumen_celular'] = (data.volumen_celular - 46.6) / 9.2
        input_dict['proteina_total'] = (data.proteina_total - 40.5) / 30.2
        
        # 2. Binary Flags
        if data.surgery == 2: input_dict['cirugia_2.0'] = 1.0
        if data.edad == 2: input_dict['edad_9'] = 1.0
        if data.lesion_quirurgica == 2: input_dict['lesion_quirurgica_2'] = 1.0
        
        # 3. Categorical Mappings (Direct One-Hot)
        mappings = {
            'dolor': (data.dolor, 'dolor_'),
            'heces': (data.heces, 'heces_rectal_'),
            'mucosas': (data.mucosas, 'membranas_mucosas_')
        }
        
        for val, prefix in mappings.values():
            if val > 1:
                key = f"{prefix}{float(val)}.0"
                if key in input_dict: input_dict[key] = 1.0
            
        df_input = pd.DataFrame([input_dict], columns=MODEL_COLUMNS)
        outcome_map = {0: "LIVED (SOBREVIVE)", 1: "DIED (MUERE)", 2: "EUTHANIZED (EUTANASIA)"}
        
        if model:
            pred_idx = model.predict(df_input)[0]
            prediction = outcome_map.get(pred_idx, f"UNKNOWN ({pred_idx})")
        else:
            prediction = "Error: Modelo V7 no cargado (Pendiente entrenamiento local)"

        report, decision = generate_ai_report(data, prediction)
        return {
            "prediction_outcome": prediction,
            "prediction_decision": decision,
            "report": report
        }
    except Exception as e:
        logger.error(f"V7 Prediction Error: {e}")
        return {"error": str(e)}

def generate_ai_report(data, result):
    if not GROQ_API_KEY: return f"Predicción: {result}.", "ANÁLISIS MANUAL"
    
    # Contexto dinámico según el riesgo
    es_critico = any(x in result.upper() for x in ["DIED", "MUERE", "EUTANASIA", "EUTHANIZED"])
    
    prompt = f"""
    ESTRATEGIA MÉDICA VETERINARIA (CÓLICO EQUINO):
    - Constantes: {data.pulso} bpm, PCV: {data.volumen_celular}%, Temp: {data.temp_rectal}°C.
    - Estado Clínico: Dolor {data.dolor}/5, Mucosas {data.mucosas}.
    - Pronóstico del Modelo V7: {result}.

    {"[!] ALERTA: PRONÓSTICO DE ALTA MORTALIDAD. Análisis de choque endotóxico requerido." if es_critico else ""}

    INSTRUCCIONES PARA EL AGENTE:
    1. Define la ruta crítica: 'CIRUGÍA RECOMENDADA' o 'TRATAMIENTO MÉDICO'.
    2. Realiza un análisis FISIOPATOLÓGICO extenso (Mínimo 4 párrafos completos).
    3. Explica la relación entre la hipovolemia (PCV), el dolor y la viabilidad intestinal.
    4. Si el pronóstico es reservado ({result}), detalla las complicaciones esperadas (SIRS, Isquemia, Fallo orgánico) y justifica científicamente la intervención sugerida.
    5. Usa terminología técnica profesional (Vet).

    RESPUESTA REQUERIDA (Formato Estricto):
    RECOMENDACIÓN: [Respuesta]
    INFORME CLÍNICO ESTRATÉGICO:
    [Contenido detallado y completo]
    """
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            full_text = response.json()['choices'][0]['message']['content']
            
            # Extracción inteligente del encabezado para el badge UI
            decision = "CONSULTA MÉDICA"
            upper_content = full_text.upper()
            if "CIRUGÍA" in upper_content.split('\n')[0] or "QUIRÚRGICA" in upper_content.split('\n')[0]:
                decision = "CIRUGÍA RECOMENDADA"
            elif "MÉDICO" in upper_content.split('\n')[0]:
                decision = "TRATAMIENTO MÉDICO"
                
            return full_text, decision
    except Exception as e:
        logger.error(f"Error en reporte IA: {e}")
    
    return f"Análisis {result}.", "CONSULTA VETERINARIA"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
