<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=120&section=header&animation=fadeIn" />
</div>

<h1 align="center">🚢 Titanic — Modelo Predictivo de Supervivencia</h1>

<h3 align="center">Machine Learning · SMOTE · Optuna · Stacking Ensemble · SHAP + LIME</h3>

<p align="center">
  Análisis completo del dataset del Titanic con estándares profesionales de Ingeniería de Datos.<br>
  Pipelines automatizados, balanceo de clases, optimización de hiperparámetros y explicabilidad.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge&logo=lightgbm&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/Optuna-4A90D9?style=for-the-badge&logo=optuna&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
</p>

---

## ⚠️ Apuntes para Entrega 2

> Estado actual del proyecto — pendientes obligatorios para la entrega.

- 📸 Imágenes y gráficos requeridos para la entrega
- 🎲 Se tiene que revisar las variables con mayor relevancia en la clase
- 📉 **Gráfico de sobreajuste OBLIGATORIO** para Árboles de Decisión y Redes Neuronales
- 🎥 **Video explicación** de cómo se realizó el agente (chatbot)
- 🌲 Análisis de overfitting también para **Random Forest**
- ⚡ **El análisis de overfitting se realiza DESPUÉS de Optuna**
- 🎲 Considerar el **Grado de Desorden** (entropía / criterio Gini)


---

## 📋 Descripción del Proyecto

El dataset del Titanic contiene información de **891 pasajeros** del famoso naufragio, con el objetivo de predecir la probabilidad de supervivencia (`Survived`: 0 = no sobrevivió, 1 = sobrevivió).

El flujo de trabajo está construido con una **arquitectura Pro de Ingeniería de Datos**, incluyendo pipelines automatizados, filtro de multicolinealidad, balanceo con SMOTE, optimización con Optuna y explicabilidad con SHAP y LIME.

---

## 🗺️ Flujo del Notebook

| # | Sección | Descripción |
|---|---------|-------------|
| 1 | **Comprensión del Negocio** | Definición del objetivo: clasificación binaria de supervivencia |
| 2 | **Comprensión de los Datos** | EDA completo, estadística descriptiva, hipótesis H2–H10, variables `FamilySize` y `Title` |
| 3 | **Limpieza y Feature Engineering** | Imputación, encoding, `CabinLetterExtractor`, `ColumnTransformer` |
| **4** | **⬇️ Generación de Modelos / Evaluación / Despliegue** | **PUNTO DE PARTIDA — ver detalle abajo** |

---

## 🔬 Sección 4 — Pasos Detallados desde el Punto de Partida

### Paso 1 — Evaluación Inicial con Validación Cruzada

Se definen **5 pipelines** con `StratifiedKFold` (5 folds) y se evalúa cada modelo con `cross_val_score`:

| Pipeline | Componentes |
|----------|-------------|
| `DecisionTree` | CorrelationFilter → SMOTE → SelectKBest → DecisionTreeClassifier |
| `KNN` | CorrelationFilter → StandardScaler → SMOTE → SelectKBest → KNeighborsClassifier |
| `SVM` | CorrelationFilter → StandardScaler → SMOTE → SelectKBest → SVC (RBF) |
| `RandomForest` | CorrelationFilter → SMOTE → SelectKBest → RandomForestClassifier (200 estimadores) |
| `LightGBM` | CorrelationFilter → SMOTE → SelectKBest → LGBMClassifier |

> **Nota:** El escalado (`StandardScaler`) solo se aplica a modelos basados en distancia o gradiente (KNN, SVM).

**Resultado:** Tabla comparativa con `Accuracy_mean` y `Accuracy_std` ordenada de mayor a menor.

---

### Paso 2 — Entrenamiento Final y Métricas sobre Test

- Entrenamiento de todos los modelos con `X_train` → diccionario `trained_models`
- Predicción sobre `X_test` y cálculo de métricas: `accuracy`, `precision`, `recall`, `F1`
- **Matrices de confusión** para cada modelo (subplots 2×3)
- **Importancia de características** extraída desde `SelectKBest` (F-Score por variable)
- `Classification Report` completo para el mejor modelo según F1-Score
- Interpretación automática de resultados con LLM (`transformers`)

---

### Paso 3 — Ajuste de Hiperparámetros con Optuna ⚡

```python
!pip install optuna
```

- Sampler: `TPESampler`
- Espacios de búsqueda definidos para cada modelo en `param_spaces`
- Función objetivo con `cross_val_score` y clonación del pipeline por trial
- Resultado almacenado en `tuned_models_optuna`
- Evaluación final con validación cruzada y tabla comparativa post-Optuna

> **🔴 Aquí se realiza el análisis de overfitting:** comparar curvas de entrenamiento vs. validación para DecisionTree, RandomForest (y Redes Neuronales si aplica).

---

### Paso 4 — Stacking Ensemble

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
```

- Base learners: mejores modelos de `tuned_models_optuna`
- Meta-modelo: `LogisticRegression`
- Ajuste de **threshold** (rango recomendado: 0.3–0.6) sobre `predict_proba`
- Comparación Stacking vs. mejor modelo individual → seleccionar el ganador
- Si el modelo individual supera al Stacking, se usa el mejor `tuned_models_optuna` entrenado con **todos los datos**

---

### Paso 5 — Explicabilidad: SHAP + LIME

```python
!pip install shap lime
```

**SHAP (global):** qué variables tienen mayor impacto positivo/negativo en la predicción  
**LIME (local):** "para ESTE pasajero específico, ¿por qué sobrevivió o no?"

**Cómo leer el output de LIME:**
- Izquierda → probabilidad asignada a cada clase
- Barras verdes → características que favorecen la supervivencia
- Barras rojas → características que reducen la probabilidad de sobrevivir

---

### Paso 6 — Guardar el Modelo + PCA (Complementario)

```python
from joblib import dump
# Guardar en Google Drive
drive.mount("/content/drive")
dump(best_model, "/content/drive/MyDrive/titanic_model.pkl")
```

- PCA a 2 componentes para visualización dimensional (opcional)
- Recomendado solo si el dataset es mayormente numérico; para datos categóricos usar MCA o FAMD

---

## 🛠️ Stack Tecnológico

<div align="center">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
</div>

<br>

| Categoría | Herramientas |
|-----------|-------------|
| **Modelos** | DecisionTree · KNN · SVM · RandomForest · LightGBM |
| **Balanceo** | SMOTE (imbalanced-learn) |
| **Selección de Features** | SelectKBest · CorrelationFilter (umbral 0.7) |
| **Optimización** | Optuna · TPESampler |
| **Ensemble** | StackingClassifier · Threshold tuning |
| **Explicabilidad** | SHAP (global) · LIME (local) |
| **Reducción dimensional** | PCA (complementario) |
| **Despliegue** | Google Colab · joblib · Google Drive |

---

## ✅ Estado del Proyecto

| Componente | Estado | Nota |
|------------|--------|------|
| EDA completo | ⏳ En curso | Hipótesis H2–H10 verificadas |
| Pipelines base (5 modelos) | ⏳ En curso | SMOTE + CorrelationFilter + SelectKBest |
| Métricas y matrices de confusión | ⏳ En curso | Accuracy, Precision, Recall, F1 |
| Optuna (ajuste hiperparámetros) | ⏳ En curso | Requiere análisis de overfitting post-Optuna |
| **Gráfico sobreajuste (DT + NN)** | ❌ Falta | **OBLIGATORIO para la entrega** |
| Stacking Ensemble | ⏳ En curso | Con threshold tuning |
| SHAP + LIME | ⏳ En curso | Global y local implementados |
| **Video explicación del agente** | ❌ Falta | **Grabar video del chatbot** |
| Serialización del modelo | ⏳ En curso | joblib en Google Drive |

---

<div align="center">

**👨‍💻 Caso de Estudio: Titanic · Ingeniería de Sistemas · Universidad del Quindío**

</div>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=90&section=footer&animation=fadeIn" />
</div>
