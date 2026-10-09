from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    source: str
    version: str
    detail: Optional[str] = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _remote_enabled() -> bool:
    required = (
        "S3_ENDPOINT",
        "S3_REGION",
        "S3_BUCKET",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
    )

    return all(os.getenv(name) for name in required)


def _create_s3_client():
    addressing_style = os.getenv(
        "S3_ADDRESSING_STYLE",
        "virtual",
    )

    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ["S3_REGION"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=Config(
            signature_version="s3v4",
            s3={
                "addressing_style": addressing_style
            },
            retries={
                "max_attempts": 3,
                "mode": "standard",
            },
        ),
    )


def _download_active_remote_model(
    *,
    cache_dir: Path,
    remote_manifest_key: str,
) -> ResolvedModel:

    client = _create_s3_client()

    bucket = os.environ["S3_BUCKET"]

    # --------------------------------------------------
    # Descargar manifest remoto
    # --------------------------------------------------

    manifest_response = client.get_object(
        Bucket=bucket,
        Key=remote_manifest_key,
    )

    manifest_bytes = manifest_response["Body"].read()

    manifest: Dict[str, Any] = json.loads(
        manifest_bytes.decode("utf-8")
    )

    if manifest.get("status") != "active":
        raise ValueError(
            "El manifest remoto no está activo: "
            f"status={manifest.get('status')!r}"
        )

    model_key = manifest.get("model_key")
    expected_sha256 = manifest.get("sha256")

    if not model_key:
        raise ValueError(
            "El manifest remoto no contiene model_key."
        )

    if not expected_sha256:
        raise ValueError(
            "El manifest remoto no contiene sha256."
        )

    # --------------------------------------------------
    # Preparar cache local
    # --------------------------------------------------

    filename = Path(str(model_key)).name

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = cache_dir / filename
    temporary = cache_dir / f"{filename}.tmp"

    # --------------------------------------------------
    # Descargar checkpoint
    # --------------------------------------------------

    client.download_file(
        Bucket=bucket,
        Key=str(model_key),
        Filename=str(temporary),
    )

    # --------------------------------------------------
    # Verificar SHA256
    # --------------------------------------------------

    actual_sha256 = _sha256_file(temporary)

    if actual_sha256.lower() != str(expected_sha256).lower():

        temporary.unlink(
            missing_ok=True
        )

        raise ValueError(
            "SHA256 inválido para el checkpoint remoto. "
            f"Esperado={expected_sha256}, "
            f"obtenido={actual_sha256}"
        )

    # --------------------------------------------------
    # Promover archivo temporal a cache válido
    # --------------------------------------------------

    temporary.replace(destination)

    return ResolvedModel(
        path=destination,
        source="remote",
        version=(
            manifest.get("model_filename")
            or filename
        ),
        detail=None,
    )


def resolve_model_path(
    *,
    local_fallback_path: Path,
    cache_dir: Optional[Path] = None,
    remote_manifest_key: Optional[str] = None,
) -> ResolvedModel:
    """
    Intenta descargar el checkpoint activo desde Railway Bucket.

    Si:
    - faltan variables S3,
    - el bucket falla,
    - active_model.json es inválido,
    - el SHA256 no coincide,

    utiliza el checkpoint local como fallback.
    """

    cache_dir = cache_dir or Path(
        os.getenv(
            "EDUFIN_MODEL_CACHE_DIR",
            "/tmp/edufin_models",
        )
    )

    remote_manifest_key = (
        remote_manifest_key
        or os.getenv(
            "EDUFIN_REMOTE_MODEL_MANIFEST",
            "models/active_model.json",
        )
    )

    # --------------------------------------------------
    # Intentar modelo remoto
    # --------------------------------------------------

    if _remote_enabled():

        try:

            return _download_active_remote_model(
                cache_dir=cache_dir,
                remote_manifest_key=remote_manifest_key,
            )

        except (
            ClientError,
            BotoCoreError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as exc:

            detail = (
                "Falló modelo remoto; "
                "usando fallback local. "
                f"Motivo: {exc}"
            )

            if local_fallback_path.exists():

                return ResolvedModel(
                    path=local_fallback_path,
                    source="local_fallback",
                    version=local_fallback_path.name,
                    detail=detail,
                )

            raise RuntimeError(
                f"{detail} "
                "Además, no existe fallback local en "
                f"{local_fallback_path}."
            ) from exc

    # --------------------------------------------------
    # Storage remoto no configurado
    # --------------------------------------------------

    if local_fallback_path.exists():

        return ResolvedModel(
            path=local_fallback_path,
            source="local_fallback",
            version=local_fallback_path.name,
            detail=(
                "Storage remoto no configurado; "
                "usando checkpoint local."
            ),
        )

    raise FileNotFoundError(
        "No hay configuración S3 completa "
        "y tampoco existe el checkpoint local en "
        f"{local_fallback_path}."
    )