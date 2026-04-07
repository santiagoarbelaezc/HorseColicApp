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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
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
    if not GROQ_API_KEY: 
        logger.error("❌ GROQ_API_KEY is MISSING in environment")
        return f"Provisional: {result}. Recomendamos consulta urgente.", "CONSULTA VETERINARIA"
    
    # Contexto dinámico
    es_critico = any(x in result.upper() for x in ["DIED", "MUERE", "EUTANASIA", "EUTHANIZED"])
    
    prompt = f"""
    ESTRATEGIA MÉDICA VETERINARIA CRÍTICA:
    - Paciente: Equino con Pulso {data.pulso} bpm, PCV {data.volumen_celular}%, Dolor {data.dolor}/5.
    - Pronóstico Modelado (V7): {result}.

    [OBJETIVO]: Generar una estrategia clínica EXTENSA y TÉCNICA.

    INSTRUCCIONES:
    1. Define 'CIRUGÍA RECOMENDADA' o 'TRATAMIENTO MÉDICO'.
    2. Desarrolla un análisis FISIOPATOLÓGICO de al menos 4 párrafos largos.
    3. Trata temas como: Desequilibrio electrolítico, Isquemia intestinal, Endotoxemia y Shock hipovolémico.
    4. Explica por qué el pronóstico es {result} y qué medidas heroicas se pueden tomar.
    5. Usa un tono de especialista de cuidados intensivos.

    FORMATO OBLIGATORIO:
    RECOMENDACIÓN: [Escribe aquí si es CIRUGÍA o MÉDICO]
    ANÁLISIS PROFUNDO:
    [Mínimo 400 palabras de análisis clínico]
    """
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant", # Cambiado a 8b para mayor velocidad y confiabilidad
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    
    try:
        logger.info("📡 Enviando consulta a Groq (8b-instant)...")
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            full_text = response.json()['choices'][0]['message']['content']
            logger.info("✅ Respuesta de IA recibida correctamente.")
            
            # Extracción robusta de la recomendación
            decision = "CONSULTA VETERINARIA"
            if "CIRUGÍA" in full_text.upper():
                decision = "CIRUGÍA RECOMENDADA"
            elif "MÉDICO" in full_text.upper():
                decision = "TRATAMIENTO MÉDICO"
            
            # Si el modelo dice que va a morir, enfatizar la urgencia quirúrgica si aplica
            if es_critico and "CIRUGÍA" in full_text.upper():
                decision = "🔴 CIRUGÍA INMEDIATA"
                
            return full_text, decision
        else:
            logger.error(f"❌ Error API Groq: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Excepción en reporte IA: {e}")
    
    return f"Análisis de emergencia para pronóstico {result}. Por favor, evalúe intervención quirúrgica inmediata si el dolor persiste.", "STADO CRÍTICO"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
