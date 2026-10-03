# EDUFIN DKT-Forget Engine v2

API de inferencia para el modelo DKT-Forget de EDUFIN.

## Estado sin modelo entrenado

La API puede iniciar aunque todavía no exista un checkpoint.

- `GET /model-status` devuelve `model_ready: false`.
- `POST /predict` devuelve HTTP 503.
- Spring debe usar una estrategia no adaptativa durante el piloto.

## Checkpoint

Cuando exista un modelo entrenado, colocar:

`motor_ia/pesos/edu_fin_dkt_forget.pth`

También se puede indicar otra ruta con la variable de entorno:

`EDUFIN_MODEL_PATH=/ruta/al/modelo.pth`

## Endpoints

### GET /model-status

Indica si el checkpoint está disponible y cargado.

### POST /predict

Ejemplo:

```json
{
  "user_id": "123",
  "interactions": [
    {
      "skill_id": 1,
      "correct": true,
      "timestamp": "2026-10-03T15:20:00Z"
    },
    {
      "skill_id": 2,
      "correct": false,
      "timestamp": "2026-10-03T15:30:00Z"
    }
  ]
}
```

Cuando el modelo esté listo devuelve las probabilidades de próxima respuesta correcta
para las 30 skills.

## Ejecutar

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
