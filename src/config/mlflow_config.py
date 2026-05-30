"""
MLflow configuration manager.

Provides a singleton that initialises the MLflow tracking server connection,
creates the experiment if it doesn't exist, and exposes a start_run()
context manager for consistent run lifecycle across the application.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

import mlflow

from src.config.settings import settings

logger = logging.getLogger(__name__)


class MLflowManager:
    """Singleton that manages MLflow tracking configuration."""

    _instance: "MLflowManager | None" = None
    _initialised: bool = False

    def __new__(cls) -> "MLflowManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialise(self) -> None:
        """Set up tracking URI and create experiment if needed."""
        if self._initialised:
            return

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        self._initialised = True
        logger.info(
            "MLflow initialised — uri=%s experiment=%s",
            settings.mlflow_tracking_uri,
            settings.mlflow_experiment_name,
        )

    @contextmanager
    def start_run(
        self, run_name: str | None = None, tags: dict[str, str] | None = None
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Start an MLflow run with the given name and tags."""
        if not self._initialised:
            self.initialise()

        # If a run is already active, we start a nested run.
        # This prevents the "Run already active" exception.
        is_nested = mlflow.active_run() is not None
        
        with mlflow.start_run(run_name=run_name, tags=tags or {}, nested=is_nested) as run:
            logger.debug(
                "MLflow %srun started: %s", 
                "nested " if is_nested else "", 
                run.info.run_id
            )
            yield run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a batch of parameters to the current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a batch of metrics to the current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Log a file artifact to the current run."""
        mlflow.log_artifact(path)


manager = MLflowManager()
