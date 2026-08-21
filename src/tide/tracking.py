"""Best-effort Trackio projection for the project-owned training record."""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Mapping
from typing import Any

from .manifest import atomic_write_json


_active_projection: "TrackioProjection | None" = None


class TrackioProjection:
    """Mirror scalar events without making Trackio the source of truth."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        *,
        mode: str,
        project: str,
        run_name: str,
        config: Mapping[str, Any],
        resume: bool,
    ) -> None:
        global _active_projection

        if _active_projection is not None:
            raise RuntimeError("only one Trackio projection may be active per process")
        if mode not in {"best-effort", "required", "off"}:
            raise ValueError(f"unknown tracking mode: {mode}")
        self.path = output_dir / "tracking.json"
        self.mode = mode
        self.project = project
        self.run_name = run_name
        self._module: Any | None = None
        self._run: Any | None = None
        self._warned = False
        self.status: dict[str, Any] = {
            "mode": mode,
            "backend": "trackio" if mode != "off" else None,
            "project": project,
            "run_name": run_name,
            "status": "disabled" if mode == "off" else "not-started",
        }
        _active_projection = self
        self._write_status()
        if mode == "off":
            return
        try:
            import trackio

            self._module = trackio
            self._run = trackio.init(
                project=project,
                name=run_name,
                config=dict(config),
                resume="must" if resume else "never",
                embed=False,
                auto_log_cpu=False,
                auto_log_gpu=False,
            )
            self.status.update(
                {
                    "status": "healthy",
                    "run_id": getattr(self._run, "id", None),
                    "version": getattr(trackio, "__version__", "unknown"),
                }
            )
            self._write_status()
        except Exception as error:
            self._degrade("init", error)

    def _write_status(self) -> None:
        atomic_write_json(self.path, self.status)

    def _degrade(self, stage: str, error: BaseException) -> None:
        self.status.update(
            {
                "status": "degraded",
                "failure_stage": stage,
                "failure_type": type(error).__name__,
            }
        )
        self._write_status()
        if self.mode == "required":
            raise RuntimeError(
                f"required Trackio operation failed during {stage}: {type(error).__name__}"
            ) from error
        if not self._warned:
            warnings.warn(
                f"Trackio degraded during {stage}; local experiment records remain active",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned = True

    def log(self, metrics: Mapping[str, int | float], *, step: int) -> None:
        if self._run is None:
            return
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as error:
            self._degrade("log", error)
            self._run = None

    def finish(self) -> None:
        global _active_projection

        if self._module is None or self._run is None:
            if _active_projection is self:
                _active_projection = None
            return
        try:
            self._module.finish()
            if self.status["status"] == "healthy":
                self.status["status"] = "finished"
                self._write_status()
        except Exception as error:
            self._degrade("finish", error)
        finally:
            self._run = None
            if _active_projection is self:
                _active_projection = None


def finish_active_projection() -> None:
    """Close the process-local projection after a training failure."""

    if _active_projection is not None:
        _active_projection.finish()
