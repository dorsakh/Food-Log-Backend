"""
Client helpers for invoking the external machine-learning inference service.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class MLServiceError(RuntimeError):
  """Raised when the external ML service returns an error."""


def call_ml_service(
  image_bytes: bytes,
  metadata: Dict[str, Any],
  *,
  url: str,
  api_key: Optional[str] = None,
  timeout: int = 30,
) -> Dict[str, Any]:
  """
  Send the uploaded image (and extracted metadata) to the remote ML API.

  Parameters
  ----------
  image_bytes:
      Raw image contents supplied by the client.
  metadata:
      EXIF-derived metadata that may enhance the ML model's accuracy.
  url:
      Fully-qualified endpoint for the ML service prediction REST API.
  api_key:
      Optional bearer token injected as ``Authorization`` header.
  timeout:
      Request timeout in seconds.
  """
  if not url:
    raise MLServiceError("ML service URL is not configured.")

  files = {
    "image": ("upload.jpg", image_bytes, "application/octet-stream"),
  }
  data = {
    "metadata": json.dumps(metadata or {}),
  }
  headers = {}
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

  try:
    response = requests.post(url, files=files, data=data, headers=headers, timeout=timeout)
  except requests.RequestException as exc:  # pragma: no cover
    raise MLServiceError(f"ML service request failed: {exc}") from exc

  if not response.ok:
    raise MLServiceError(
      f"ML service responded with {response.status_code}: {response.text}"
    )

  try:
    return response.json()
  except ValueError as exc:  # pragma: no cover
    raise MLServiceError("ML service did not return JSON.") from exc


__all__ = ["call_ml_service", "MLServiceError"]
