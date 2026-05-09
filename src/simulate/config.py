import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ComponentConfig(BaseModel):
    """Base configuration for any component in the simulation."""

    dt: float = Field(..., gt=0, description="Sample time / update period of the component")


class PlantConfig(ComponentConfig):
    """Base configuration for the plant."""


class SensorConfig(ComponentConfig):
    """Base configuration for the sensor."""


class EstimatorConfig(ComponentConfig):
    """Base configuration for the estimator."""


class ControllerConfig(ComponentConfig):
    """Base configuration for the controller."""


class LinearPlantConfig(PlantConfig):
    """Configuration for the linear plant."""

    a: list[list[float]] = Field(..., description="System matrix A")
    b: list[list[float]] = Field(..., description="Input matrix B")
    c: list[list[float]] = Field(..., description="Output matrix C")
    d: list[list[float]] = Field(..., description="Feedthrough matrix D")


class GaussianSensorConfig(SensorConfig):
    """Configuration for a sensor with Gaussian noise."""

    std_dev: float = Field(default=0.0, ge=0, description="Standard deviation of the Gaussian noise")


class IdentityEstimatorConfig(EstimatorConfig):
    """Configuration for an identity estimator."""


class PIDControllerConfig(ControllerConfig):
    """Configuration for the PID controller."""

    kp: list[list[float]] = Field(..., description="Proportional gain matrix")
    ki: list[list[float]] = Field(..., description="Integral gain matrix")
    kd: list[list[float]] = Field(..., description="Derivative gain matrix")


class SimulationConfig[P: PlantConfig, S: SensorConfig, E: EstimatorConfig, C: ControllerConfig](BaseModel):
    """Root configuration object for the entire simulation."""

    plant: P
    sensor: S
    estimator: E
    controller: C
    t_end: float = Field(..., gt=0, description="End time of the simulation")

    @model_validator(mode="after")
    def validate_sample_times(self) -> "SimulationConfig[P, S, E, C]":
        """Validate that all component sample times are integer multiples of the plant's base dt."""
        base_dt = self.plant.dt

        for component_name in ["sensor", "estimator", "controller"]:
            component = getattr(self, component_name)
            dt = component.dt
            ratio = dt / base_dt
            if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                msg = f"{component_name.capitalize()} dt ({dt}) must be an integer multiple of plant dt ({base_dt})"
                raise ValueError(msg)

        return self


def load_config(filepath: str | Path) -> SimulationConfig:
    """Load and validate a YAML configuration file."""
    with Path(filepath).open() as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)

    # Note: In a real system, we'd need a discriminator or custom logic to instantiate
    # the correct subclasses based on the yaml. For now, this serves the basic structure.
    return SimulationConfig(**raw_config)
