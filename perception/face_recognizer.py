"""
perception/face_recognizer.py
------------------------------
Stage 3a — Face Recognition

Responsibilities:
  1. Load enrolled face embeddings from disk at startup
  2. For each incoming frame, detect faces and compare against enrolled DB
  3. Return FaceDetection(person_id, confidence, bbox, is_known)
  4. Gracefully fall back to anonymous mode when:
     - No face detected in frame
     - Face detected but similarity < threshold (unknown person)
     - DeepFace not installed (anonymous_only mode)
     - use_case has identity_required: false

DEMO VIDEO NOTE:
  For downloaded housekeeping videos, faces won't match anyone enrolled.
  That's correct behavior — FaceDetection.is_known = False, person_id = None.
  The system runs in anonymous mode and still validates behaviors.
  Only the identity-bound use case (use case 2) needs enrolled faces.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .enriched_frame import FaceDetection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Enrolled person record (loaded from YAML)
# ─────────────────────────────────────────────

class EnrolledPerson:
    def __init__(self, data: dict) -> None:
        self.id               = data["id"]
        self.name             = data["name"]
        self.role             = data.get("role", "unknown")
        self.embedding_path   = Path(data["embedding_path"])
        self.allowed_zones    = data.get("allowed_zones", [])
        self.restricted_zones = data.get("restricted_zones", [])
        self.use_cases        = data.get("use_cases", [])
        self.active           = data.get("active", True)
        self.embedding: Optional[np.ndarray] = None  # loaded lazily

    def load_embedding(self) -> bool:
        """Load face embedding from .npy file. Returns False if file missing."""
        if not self.embedding_path.exists():
            logger.warning("Embedding file not found for %s: %s", self.id, self.embedding_path)
            return False
        self.embedding = np.load(str(self.embedding_path))
        logger.debug("Loaded embedding for %s (shape: %s)", self.id, self.embedding.shape)
        return True


# ─────────────────────────────────────────────
# Face Recognizer
# ─────────────────────────────────────────────

class FaceRecognizer:
    """
    Wraps DeepFace to provide face detection and recognition.
    Falls back gracefully to anonymous mode when DeepFace is unavailable
    or no enrolled persons are configured.

    Usage
    -----
        recognizer = FaceRecognizer("config/identities.yaml")
        face = recognizer.recognize(frame)
        print(face.display_id)   # "staff_member_01" or "unknown"
    """

    def __init__(self, config_path: str = "config/identities.yaml") -> None:
        self._config_path       = Path(config_path)
        self._enrolled:         list[EnrolledPerson] = []
        self._threshold:        float = 0.65
        self._model_name:       str   = "VGG-Face"
        self._detector_backend: str   = "opencv"
        self._enforce_detection: bool = False
        self._anonymous_label:  str   = "unknown"
        self._deepface_available: bool = False

        self._load_config()
        self._check_deepface()
        self._load_embeddings()

    # ── public API ────────────────────────────────────────

    def recognize(self, frame: np.ndarray) -> FaceDetection:
        """
        Main entry point. Given a BGR frame, returns a FaceDetection.

        Always returns a FaceDetection — never raises.
        If anything fails, returns anonymous result.
        """
        t0 = time.monotonic()

        if not self._deepface_available:
            return self._anonymous_result(reason="DeepFace not available")

        if not self._enrolled:
            return self._anonymous_result(reason="No enrolled persons in database")

        try:
            return self._run_recognition(frame)
        except Exception as exc:
            logger.debug("Face recognition failed (will use anonymous): %s", exc)
            return self._anonymous_result(reason=str(exc))

    def get_person(self, person_id: str) -> Optional[EnrolledPerson]:
        """Look up an enrolled person by ID."""
        return next((p for p in self._enrolled if p.id == person_id), None)

    def enroll_from_frame(
        self,
        frame: np.ndarray,
        person_id: str,
        person_name: str,
        save_dir: str = "data/enrolled_faces",
    ) -> bool:
        """
        Extract face embedding from a frame and save it.
        Used by tools/enroll_face.py.
        Returns True on success.
        """
        if not self._deepface_available:
            logger.error("Cannot enroll — DeepFace not installed.")
            return False

        try:
            from deepface import DeepFace
            embedding_data = DeepFace.represent(
                img_path=frame,
                model_name=self._model_name,
                detector_backend=self._detector_backend,
                enforce_detection=True,   # must find a face for enrollment
            )
            if not embedding_data:
                logger.error("No face detected in enrollment frame for %s", person_id)
                return False

            embedding = np.array(embedding_data[0]["embedding"])
            save_path = Path(save_dir) / f"{person_id}.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), embedding)

            logger.info("Enrolled %s (%s) → %s", person_id, person_name, save_path)
            return True

        except Exception as exc:
            logger.error("Enrollment failed for %s: %s", person_id, exc)
            return False

    # ── internals ─────────────────────────────────────────

    def _load_config(self) -> None:
        if not self._config_path.exists():
            logger.warning("identities.yaml not found at %s — running in anonymous mode", self._config_path)
            return

        with self._config_path.open() as f:
            raw = yaml.safe_load(f)

        settings = raw.get("settings", {})
        self._threshold         = settings.get("similarity_threshold", 0.65)
        self._model_name        = settings.get("model", "VGG-Face")
        self._detector_backend  = settings.get("detector", "opencv")
        self._enforce_detection = settings.get("enforce_detection", False)
        self._anonymous_label   = settings.get("anonymous_label", "unknown")

        persons_raw = raw.get("persons", [])
        self._enrolled = [EnrolledPerson(p) for p in persons_raw if p.get("active", True)]
        logger.info("Loaded %d enrolled person(s) from config", len(self._enrolled))

    def _check_deepface(self) -> None:
        try:
            import deepface  # noqa: F401
            self._deepface_available = True
            logger.info("DeepFace available — face recognition enabled")
        except ImportError:
            self._deepface_available = False
            logger.warning(
                "DeepFace not installed. Running in anonymous mode.\n"
                "  Install with: pip install deepface\n"
                "  This is fine for housekeeping use case (identity not required)."
            )

    def _load_embeddings(self) -> None:
        if not self._deepface_available or not self._enrolled:
            return

        loaded = 0
        for person in self._enrolled:
            if person.load_embedding():
                loaded += 1

        logger.info("Loaded embeddings for %d/%d enrolled person(s)", loaded, len(self._enrolled))

    def _run_recognition(self, frame: np.ndarray) -> FaceDetection:
        """Core DeepFace recognition logic."""
        from deepface import DeepFace

        # Step 1: detect face(s) in frame and get embedding
        try:
            embedding_data = DeepFace.represent(
                img_path=frame,
                model_name=self._model_name,
                detector_backend=self._detector_backend,
                enforce_detection=self._enforce_detection,
            )
        except Exception:
            return self._anonymous_result(reason="No face detected in frame")

        if not embedding_data:
            return self._anonymous_result(reason="No face detected in frame")

        # Use the highest-confidence face if multiple detected
        best_face_data = embedding_data[0]
        query_embedding = np.array(best_face_data["embedding"])
        facial_area = best_face_data.get("facial_area", {})
        bbox = (
            facial_area.get("x", 0),
            facial_area.get("y", 0),
            facial_area.get("w", 0),
            facial_area.get("h", 0),
        ) if facial_area else None

        # Step 2: compare against all enrolled embeddings
        best_person  = None
        best_similarity = 0.0

        for person in self._enrolled:
            if person.embedding is None:
                continue
            similarity = self._cosine_similarity(query_embedding, person.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person

        # Step 3: threshold check
        if best_person and best_similarity >= self._threshold:
            logger.debug(
                "Recognized: %s (similarity=%.3f)", best_person.id, best_similarity
            )
            return FaceDetection(
                person_id=best_person.id,
                person_name=best_person.name,
                confidence=best_similarity,
                bbox=bbox,
                is_known=True,
            )

        if best_person:
            logger.debug(
                "Face detected but below threshold (best=%.3f < %.3f) — anonymous",
                best_similarity, self._threshold,
            )
        return self._anonymous_result(reason="Below similarity threshold", bbox=bbox)

    def _anonymous_result(
        self,
        reason: str = "",
        bbox: Optional[tuple] = None,
    ) -> FaceDetection:
        if reason:
            logger.debug("Anonymous mode: %s", reason)
        return FaceDetection(
            person_id=None,
            person_name=None,
            confidence=0.0,
            bbox=bbox,
            is_known=False,
            anonymous_label=self._anonymous_label,
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors. Returns 0.0–1.0."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
