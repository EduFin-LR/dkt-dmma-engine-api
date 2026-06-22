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

# 1. Definimos los "Contratos" (Data Transfer Objects)
class SolicitudPrediccion(BaseModel):
    user_id: str = Field(..., description="ID del usuario en Spring Boot")
    secuencia_interacciones: List[int] = Field(..., description="Historial de IDs codificados (Habilidad + Acierto/Error)")
    habilidad_objetivo: int = Field(..., description="El ID del tema financiero que queremos evaluar (Ej. 2 para Presupuesto)")
    dias_inactividad: float = Field(..., description="Tiempo transcurrido desde su última sesión en días")

class RespuestaPrediccion(BaseModel):
    user_id: str
    habilidad_objetivo: int
    probabilidad_base_dkt: float
    probabilidad_final_dmma: float
    nivel_recomendado: str

# 2. Evento de Arranque
@app.on_event("startup")
def cargar_modelo():
    global modelo_predictivo
    print("Iniciando el motor y cargando la memoria de la IA...")
    
    # IMPORTANTE: Reemplaza 120 por el número exacto de habilidades de tu entrenamiento
    modelo_predictivo = DKTModel(num_skills=12368, embed_dim=32, hidden_dim=64)
    modelo_predictivo.load_state_dict(torch.load("motor_ia/pesos/modelo_dkt_finanzas.pth", map_location=torch.device('cpu')))
    modelo_predictivo.eval()
    print("¡Modelo cargado y listo para la inferencia!")

# 3. La Matemática del Olvido (Tu Innovación)
def aplicar_curva_olvido_dmma(probabilidad_base: float, dias_transcurridos: float) -> float:
    if dias_transcurridos <= 0:
        return probabilidad_base
        
    fuerza_memoria = probabilidad_base + 0.01 
    retencion_actual = math.exp(-(dias_transcurridos * 0.1) / fuerza_memoria)
    probabilidad_final = probabilidad_base * retencion_actual
    
    return round(probabilidad_final, 4)

# 4. El Endpoint Principal (Donde ocurre la magia)
@app.post("/predecir-nivel", response_model=RespuestaPrediccion)
def predecir_nivel_conocimiento(solicitud: SolicitudPrediccion):
    if not modelo_predictivo:
        raise HTTPException(status_code=500, detail="El modelo DKT no está en memoria.")
    
    try:
        # A. Preparamos el array de Python para que PyTorch lo entienda (Tensor)
        # Le añadimos unos corchetes extra [ ] para simular el "batch_size" de 1
        tensor_historial = torch.tensor([solicitud.secuencia_interacciones], dtype=torch.long)
        
        # B. Inferencia Estática (DKT)
        with torch.no_grad(): # Desactivamos el entrenamiento para ahorrar RAM y ganar velocidad
            predicciones = modelo_predictivo(tensor_historial)
            
            # predicciones tiene 3 dimensiones: [lote, tiempo, habilidades]
            # Queremos: lote 0, el ÚLTIMO paso de tiempo (-1), y la habilidad específica que pide Java
            prob_base = predicciones[0, -1, solicitud.habilidad_objetivo].item()
            prob_base = round(prob_base, 4)
            
        # C. Inferencia Dinámica Temporal (DMMA)
        prob_final = aplicar_curva_olvido_dmma(prob_base, solicitud.dias_inactividad)
        
        # D. Motor de Reglas Simple para ayudar a Spring Boot
        if prob_final >= 0.75:
            dificultad = "Nivel 3 (Avanzado)"
        elif prob_final >= 0.40:
            dificultad = "Nivel 2 (Intermedio)"
        else:
            dificultad = "Nivel 1 (Repaso / Fácil)"
            
        return RespuestaPrediccion(
            user_id=solicitud.user_id,
            habilidad_objetivo=solicitud.habilidad_objetivo,
            probabilidad_base_dkt=prob_base,
            probabilidad_final_dmma=prob_final,
            nivel_recomendado=dificultad
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el procesamiento de tensores: {str(e)}")