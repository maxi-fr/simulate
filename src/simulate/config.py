import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ComponentConfig(BaseModel):
    """Base configuration for any component in the simulation."""

    dt: float = Field(..., gt=0, description="Sample time / update period of the component")


class LinearPlantConfig(ComponentConfig):
    """Configuration for the linear plant."""

    a: list[list[float]] = Field(..., description="System matrix A")
    b: list[list[float]] = Field(..., description="Input matrix B")
    c: list[list[float]] = Field(..., description="Output matrix C")
    d: list[list[float]] = Field(..., description="Feedthrough matrix D")


class PIDControllerConfig(ComponentConfig):
    """Configuration for the PID controller."""

    kp: list[list[float]] = Field(..., description="Proportional gain matrix")
    ki: list[list[float]] = Field(..., description="Integral gain matrix")
    kd: list[list[float]] = Field(..., description="Derivative gain matrix")


class SimulationConfig(BaseModel):
    """Root configuration object for the entire simulation."""

    plant: LinearPlantConfig
    controller: PIDControllerConfig
    t_end: float = Field(..., gt=0, description="End time of the simulation")

    @model_validator(mode="after")
    def validate_sample_times(self) -> "SimulationConfig":
        """Validate that all component sample times are integer multiples of the plant's base dt."""
        base_dt = self.plant.dt

        # Check controller
        controller_dt = self.controller.dt
        ratio = controller_dt / base_dt
        if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
            msg = f"Controller dt ({controller_dt}) must be an integer multiple of plant dt ({base_dt})"
            raise ValueError(msg)

        return self


def load_config(filepath: str | Path) -> SimulationConfig:
    """Load and validate a YAML configuration file."""
    with Path(filepath).open() as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)

    return SimulationConfig(**raw_config)
