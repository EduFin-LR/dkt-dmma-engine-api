from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from motor_ia.dkt_forget_final import (
    EDUFIN_NUM_SKILLS,
    DKTForgetModel,
    load_checkpoint,
    predict_mastery,
)

from motor_ia.model_storage import (
    resolve_model_path,
)


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    BASE_DIR
    / "motor_ia"
    / "pesos"
    / "edu_fin_dkt_forget.pth"
)

MODEL_PATH = Path(
    os.getenv(
        "EDUFIN_MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )
)


modelo_predictivo: Optional[DKTForgetModel] = None
checkpoint: Optional[dict] = None

model_load_error: Optional[str] = None
model_source: Optional[str] = None
model_version: Optional[str] = None
model_detail: Optional[str] = None


class Interaccion(BaseModel):

    skill_id: int = Field(
        ...,
        ge=1,
        le=EDUFIN_NUM_SKILLS,
    )

    correct: bool

    timestamp: datetime


class SolicitudPrediccion(BaseModel):

    user_id: str = Field(
        ...,
        min_length=1,
    )

    interactions: List[Interaccion] = Field(
        ...,
        min_length=1,
    )


class RespuestaPrediccion(BaseModel):

    user_id: str

    model_ready: bool

    mastery: Dict[str, float]


class EstadoModelo(BaseModel):

    model_ready: bool

    model_name: str

    num_skills: int

    model_source: Optional[str] = None

    model_version: Optional[str] = None

    detail: Optional[str] = None


def cargar_modelo_si_existe() -> None:

    global modelo_predictivo
    global checkpoint
    global model_load_error
    global model_source
    global model_version
    global model_detail

    modelo_predictivo = None
    checkpoint = None

    model_load_error = None
    model_source = None
    model_version = None
    model_detail = None

    try:

        # --------------------------------------------------
        # Resolver modelo:
        #
        # 1. Bucket remoto
        # 2. Fallback local
        # --------------------------------------------------

        resolved = resolve_model_path(
            local_fallback_path=MODEL_PATH,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------
        # Cargar checkpoint
        # --------------------------------------------------

        modelo_predictivo, checkpoint = (
            load_checkpoint(
                resolved.path,
                device=device,
            )
        )

        num_skills = int(
            checkpoint["num_skills"]
        )

        # --------------------------------------------------
        # Validación EDUFIN
        # --------------------------------------------------

        if num_skills != EDUFIN_NUM_SKILLS:

            raise ValueError(
                f"El checkpoint contiene "
                f"{num_skills} skills; "
                f"EDUFIN requiere "
                f"{EDUFIN_NUM_SKILLS}."
            )

        # --------------------------------------------------
        # Metadata de modelo cargado
        # --------------------------------------------------

        model_source = resolved.source

        model_version = resolved.version

        model_detail = resolved.detail

        print(
            "[DKT-FORGET] "
            "Modelo cargado correctamente "
            f"({num_skills} skills, "
            f"source={model_source}, "
            f"version={model_version})."
        )

        if model_detail:

            print(
                "[DKT-FORGET] INFO: "
                f"{model_detail}"
            )

    except Exception as exc:

        modelo_predictivo = None

        checkpoint = None

        model_load_error = str(exc)

        model_source = None

        model_version = None

        model_detail = None

        print(
            "[DKT-FORGET] ERROR: "
            "No se pudo cargar el checkpoint: "
            f"{exc}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):

    cargar_modelo_si_existe()

    yield


app = FastAPI(

    title="EDUFIN DKT-Forget Engine",

    description=(
        "Microservicio de inferencia "
        "DKT-Forget para las 30 skills "
        "financieras de EDUFIN."
    ),

    version="2.1.0",

    lifespan=lifespan,
)


@app.get(
    "/model-status",
    response_model=EstadoModelo,
)
def obtener_estado_modelo() -> EstadoModelo:

    ready = (
        modelo_predictivo is not None
        and checkpoint is not None
    )

    return EstadoModelo(

        model_ready=ready,

        model_name="dkt_forget",

        num_skills=EDUFIN_NUM_SKILLS,

        model_source=model_source,

        model_version=model_version,

        detail=(
            model_detail
            if ready
            else model_load_error
        ),
    )


@app.post(
    "/predict",
    response_model=RespuestaPrediccion,
)
def predecir_mastery(
    solicitud: SolicitudPrediccion,
) -> RespuestaPrediccion:

    if (
        modelo_predictivo is None
        or checkpoint is None
    ):

        raise HTTPException(

            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),

            detail=(
                "El modelo DKT-Forget todavía "
                "no está entrenado/cargado. "
                "Utiliza la estrategia "
                "no adaptativa durante el piloto."
            ),
        )

    # --------------------------------------------------
    # Orden cronológico
    # --------------------------------------------------

    interactions = sorted(
        solicitud.interactions,
        key=lambda item: item.timestamp,
    )

    payload = [

        {
            "skill_id": item.skill_id,
            "correct": item.correct,
            "timestamp": item.timestamp,
        }

        for item in interactions
    ]

    # --------------------------------------------------
    # Inferencia
    # --------------------------------------------------

    try:

        mastery = predict_mastery(
            modelo_predictivo,
            checkpoint,
            payload,
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as exc:

        raise HTTPException(

            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),

            detail=str(exc),

        ) from exc

    except Exception as exc:

        raise HTTPException(

            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),

            detail=(
                "Error interno durante "
                "la inferencia DKT-Forget."
            ),

        ) from exc

    # --------------------------------------------------
    # Serializar mastery
    # --------------------------------------------------

    mastery_json = {

        str(skill_id): float(probability)

        for skill_id, probability
        in mastery.items()
    }

    return RespuestaPrediccion(

        user_id=solicitud.user_id,

        model_ready=True,

        mastery=mastery_json,
    )