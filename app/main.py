from fastapi import FastAPI
import torch

# Importamos la arquitectura de tu modelo desde la otra carpeta
from motor_ia.dkt_model import DKTModel

# 1. Inicializamos la aplicación FastAPI
app = FastAPI(
    title="Motor Predictivo DKT & DMMA",
    description="Microservicio de IA para la plataforma adaptativa de educación financiera",
    version="1.0.0"
)

# Variable global para mantener el modelo en memoria
modelo_predictivo = None

# 2. Evento de Arranque: Esto corre UNA SOLA VEZ cuando enciendes el servidor
@app.on_event("startup")
def cargar_modelo():
    global modelo_predictivo
    print("Iniciando el motor y cargando la memoria de la IA...")
    
    # IMPORTANTE: Los parámetros deben ser exactamente los mismos que usaste en Colab
    # Si num_skills fue 100, embed_dim 32 y hidden_dim 64, pon los mismos aquí.
    num_habilidades_entrenadas = 12368 # Reemplaza con el número que te dio en Colab
    
    modelo_predictivo = DKTModel(num_skills=num_habilidades_entrenadas, embed_dim=32, hidden_dim=64)
    
    # Cargamos el archivo .pth
    modelo_predictivo.load_state_dict(torch.load("motor_ia/pesos/modelo_dkt_finanzas.pth", map_location=torch.device('cpu')))
    
    # Lo ponemos en modo de "Evaluación/Inferencia" (para que no siga entrenando)
    modelo_predictivo.eval()
    print("¡Modelo cargado y listo para recibir alumnos!")

# 3. Endpoint de prueba para verificar que el servidor vive
@app.get("/")
def health_check():
    return {"status": "ok", "mensaje": "El Motor Predictivo está en línea"}