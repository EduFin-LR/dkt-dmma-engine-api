from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch
import math

# Importamos la arquitectura de tu modelo
from motor_ia.dkt_model import DKTModel

app = FastAPI(
    title="Motor Predictivo DKT & DMMA",
    description="Microservicio de inferencia para evaluar la retención de conocimiento financiero",
    version="1.0.0"
)

modelo_predictivo = None

# 🔥 1. DICCIONARIO DE HOMÓNIMOS (Mapeo de tus 12 Skills locales al espacio de ASSISTments)
# Reemplaza estos números muestra (101, 102...) por los índices reales de tu entrenamiento.
MAPEO_HOMONIMOS = {
    1: 101,   # Finanzas Personales Básicas - Ingresos y Gastos
    2: 102,   # Gastos Hormiga y Vampiro
    3: 103,   # Ahorro e Inflación
    4: 104,   # Costo de Oportunidad y Estafas
    5: 105,   # Ciberseguridad Bancaria
    6: 106,   # Salud Financiera (Superávit)
    7: 201,   # Módulo 2 - Introducción al Crédito
    8: 202,   # Historial Crediticio (Infocorp)
    9: 203,   # Tarjetas de Crédito y Tasas
    10: 204,  # Financiamiento Sostenible
    11: 205,  # Inversiones y Riesgo
    12: 206   # Planificación de Futuro
}

# 2. Definimos los "Contratos" (Data Transfer Objects)
class SolicitudPrediccion(BaseModel):
    user_id: str = Field(..., description="ID del usuario en Spring Boot")
    secuencia_interacciones: List[int] = Field(..., description="Historial de IDs codificados (Habilidad + Acierto/Error)")
    habilidad_objetivo: int = Field(..., description="El ID del tema financiero local (1 al 12)")
    dias_inactividad: float = Field(..., description="Tiempo transcurrido desde su última sesión en días")

class RespuestaPrediccion(BaseModel):
    user_id: str
    habilidad_objetivo: int
    probabilidad_base_dkt: float
    probabilidad_final_dmma: float
    nivel_recommended: str

# 3. Evento de Arranque
@app.on_event("startup")
def cargar_modelo():
    global modelo_predictivo
    print("Iniciando el motor y cargando la memoria de la IA...")
    try:
        # Inicialización basada en las dimensiones del preentrenamiento de ASSISTments 2009
        modelo_predictivo = DKTModel(num_skills=12368, embed_dim=32, hidden_dim=64)
        modelo_predictivo.load_state_dict(torch.load("motor_ia/pesos/modelo_dkt_finanzas.pth", map_location=torch.device('cpu')))
        modelo_predictivo.eval()
        print("¡Modelo cargado y listo para la inferencia!")
    except Exception as e:
        print(f"❌ Error crítico cargando los pesos .pth: {str(e)}")

# 4. La Matemática del Olvido (DMMA)
def aplicar_curva_olvido_dmma(probabilidad_base: float, dias_transcurridos: float) -> float:
    if dias_transcurridos <= 0:
        return probabilidad_base
        
    fuerza_memoria = probabilidad_base + 0.01 
    retencion_actual = math.exp(-(dias_transcurridos * 0.1) / fuerza_memoria)
    probabilidad_final = probabilidad_base * retencion_actual
    
    return round(probabilidad_final, 4)

# 5. El Endpoint Principal 
@app.post("/predecir-nivel", response_model=RespuestaPrediccion)
def predecir_nivel_conocimiento(solicitud: SolicitudPrediccion):
    if not modelo_predictivo:
        raise HTTPException(status_code=500, detail="El modelo DKT no está en memoria.")
    
    # 🔍 TRADUCCIÓN DE HOMÓNIMO
    if solicitud.habilidad_objetivo not in MAPEO_HOMONIMOS:
        raise HTTPException(status_code=400, detail=f"La habilidad local {solicitud.habilidad_objetivo} no está mapeada en el diccionario de homónimos.")
    
    id_habilidad_modelo = MAPEO_HOMONIMOS[solicitud.habilidad_objetivo]
    
    try:
        # 🛡️ PROTECCIÓN CONTRA HISTORIAL VACÍO
        if not solicitud.secuencia_interacciones:
            prob_base = 0.5000  # Asignamos un comportamiento neutral por defecto
            print(f"[IA] Usuario {solicitud.user_id} no registra historial. Retornando probabilidad base por defecto.")
        else:
            # Preparamos el array de Python para que PyTorch lo entienda (Tensor)
            tensor_historial = torch.tensor([solicitud.secuencia_interacciones], dtype=torch.long)
            
            # Inferencia Estática (DKT)
            with torch.no_grad():
                predicciones = modelo_predictivo(tensor_historial)
                # Extraemos la probabilidad usando el ID que el modelo ASSISTments sí entiende
                prob_base = predicciones[0, -1, id_habilidad_modelo].item()
                prob_base = round(prob_base, 4)
            
        # Inferencia Dinámica Temporal (DMMA)
        prob_final = aplicar_curva_olvido_dmma(prob_base, solicitud.dias_inactividad)
        
        # Motor de Reglas Adaptativo para Spring Boot
        if prob_final >= 0.75:
            dificultad = "Nivel 3 (Avanzado)"
        elif prob_final >= 0.40:
            dificultad = "Nivel 2 (Intermedio)"
        else:
            dificultad = "Nivel 1 (Repaso / Fácil)"
            
        return RespuestaPrediccion(
            user_id=solicitud.user_id,
            habilidad_objetivo=solicitud.habilidad_objetivo, # Retornamos el ID local para que Spring Boot lo entienda
            probabilidad_base_dkt=prob_base,
            probabilidad_final_dmma=prob_final,
            nivel_recommended=dificultad
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el procesamiento de tensores en PyTorch: {str(e)}")