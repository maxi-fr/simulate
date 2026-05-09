# Whitepaper: Modular Python Framework for Control System Simulation

#### Executive Summary

This whitepaper outlines the software development process and architectural design for a custom, modular Python framework dedicated to control system simulation. Moving away from monolithic proprietary platforms, this framework emphasizes simplicity, object-oriented principles, and robust software engineering. It incorporates modern tooling standards, utilizing YAML with dynamic class loading for configuration parsing, a standardized `Logger` for multi-format data persistence, integrated Continuous/Discrete plant dynamics via custom integrators, and a scalable multiprocessing architecture for batch experimentation.

## 1. Object-Oriented Architectural Paradigm

To ensure a modular and extensible environment, the framework adopts a block-based Object-Oriented Programming (OOP) paradigm. Each physical or computational element within the feedback loop is encapsulated within a dedicated Python class. This separation guarantees that localized logic, state, and parameters remain isolated.

### 1.1. The Plant

The Plant encapsulates the mathematical model of the physical system. To fulfill advanced design constraints, the Plant class is engineered to support both discrete and continuous dynamics.

*   **Discrete Time:** Modeled using difference equations where the state advances as `x_k+1 = f(t_k, x_k, u_k, w_k)`.
*   **Continuous Time:** The Plant class utilizes custom-coded numerical integrators (e.g., internal Runge-Kutta or Euler schemes). The standard `update(t_k, u_k)` method dynamically routes the control input through the selected custom integrator to advance the internal state efficiently without the overhead of external solver initializations per step.

### 1.2. The Sensor, Estimator, and Controller

To accurately model hardware limitations and computational realities, the remaining feedback blocks act as strict, stateful filters:

*   **Sensor:** Ingests the true physical output `y_k` from the Plant and applies transformations (e.g., Gaussian noise, latency) to generate a measured output `ym_k`.
*   **Estimator:** Reconstructs the unmeasured state `x_hat_k` using the noisy measurement `ym_k` and the known control input `u_k`.
*   **Controller:** Executes the core logic (e.g., PID, MPC) to compute the optimal control effort `u_k` based on the error between a desired reference trajectory and the estimated state.

## 2. The Simulation Orchestrator

A centralized `Simulation` class manages the chronological progression of time and data routing. To support physically accurate **multi-rate systems**, the orchestrator implements a "Base Tick" architecture driven by the physical plant.

During initialization, the simulation's fundamental time step is automatically set to match the Plant's configured update period. To guarantee synchronous execution and avoid floating-point time drift, the configuration manager strictly enforces that the update periods of all other modules (Sensors, Estimators, Controllers) are integer multiples of this base period.

During the execution loop, the orchestrator passes the current simulation time (`t_k`) to each block. Modules independently evaluate this timestamp against their configured internal sample times. If an update is due, the module executes its logic; otherwise, it performs a **Zero-Order Hold (ZOH)**, bypassing computation and returning its previously held state.

The execution sequence rigorously follows causal logic to prevent algebraic loops:

1.  **Reference Generation:** Determine the setpoint for the current time step (`t_k`).
2.  **Measurement:** Read the Plant's output via the Sensor at time `t_k`.
3.  **Estimation:** Calculate the state estimate using the measurement and previous input.
4.  **Control:** Compute the new control action via the Controller at time `t_k`.
5.  **Actuation:** Apply the control action to update the Plant.

## 3. Configuration Management

A fundamental requirement of this framework is the strict separation of code and experimental parameters. The framework adopts a configuration-driven architecture using YAML and dynamic class resolution.

#### Configuration Toolchain: YAML + Dynamic Loading

While `PyYAML` (specifically `yaml.safe_load()`) parses the human-readable configuration files, hardcoding component classes limits extensibility. Instead, the YAML dictates the specific component to instantiate by providing a `class_path` string (e.g., `simulate.plant.LinearPlant`). The central orchestrator dynamically imports these classes and passes their respective parameter dictionaries to a `from_config` factory method required on each component. This approach relies on normal class constructors and manual parameter extraction and type validation within the `from_config` methods, giving developers maximum flexibility to define their components without enforcing heavy external schema dependencies.

## 4. Data Logging and Persistence

Data generated during the simulation loop must be efficiently captured for post-processing. A dedicated, centralized `Logger` class handles file I/O separately from the core loop.

The `Logger` implements a standardized interface supporting multiple export formats to balance accessibility and performance:

*   **CSV Export:** Utilizes Pandas to flatten the data into a human-readable, universally accessible `.csv` file, ideal for quick visual inspection.
*   **NumPy Archive (.npz):** Provides a high-performance, compressed binary format using `numpy.savez`. This is the preferred method for large numerical arrays and multi-dimensional state vectors, as it preserves exact floating-point precision without massive file size overhead.

### 4.1. Component-Driven Dual Logging Architecture

To ensure comprehensive data collection without tightly coupling the central simulation orchestrator to the internal logic of individual components, the framework employs a component-driven dual logging architecture. Standardized signal vectors—such as the plant state (`x`), control effort (`u`), estimated state (`x_hat`), and sensor measurements (`y_mea`)—are inherent to every control loop and are logged universally across all simulations.

However, advanced algorithms require tracking specialized internal variables. To bridge the gap between dynamic internal states and strict type-safety, **components must define their logging schema using Pydantic models**.

When the orchestrator calls a component's step function, the method returns a tuple containing both the primary operational output and an instance of its defined Pydantic log model. The orchestrator unpacks these tuples, aggregating the universal signals into a standard dictionary snapshot, while capturing the strictly-typed component logs into a secondary snapshot. Both are appended to accumulation lists. Upon termination, these lists are converted into structured formats (such as Pandas DataFrames) and flushed to disk. Using Pydantic for internal logs prevents mid-simulation schema changes (e.g., dynamically adding a key to a dictionary) that would otherwise crash the final DataFrame compilation.

> **Note on Future Performance Optimization:** Currently, the framework accumulates log data using Python lists and dictionaries during the simulation loop. While flexible for initial development, object-referenced lists carry memory overhead. In future iterations, to support exceptionally high-frequency or long-horizon simulations without memory bloat, the logging backbone will be migrated to use pre-allocated, fixed-size NumPy arrays populated via index tracking.

## 5. Batch Execution and Experimentation

For robustness analysis and controller tuning, the framework incorporates an `ExperimentManager` to conduct batch parameter sweeps. To bypass Python's Global Interpreter Lock (GIL), the framework utilizes process-based concurrency via `multiprocessing.Pool`.

To guarantee peak performance and adhere to strict pickling constraints, the orchestrator only passes the lightweight configuration dictionaries to the worker processes. The worker processes instantiate the simulation components locally via the `class_path` properties, run the loop, and **write their resulting aggregated data directly to disk as isolated `.npz` files.** The workers then return only a lightweight completion status to the parent process. This architecture completely eliminates Inter-Process Communication (IPC) bottlenecks that would otherwise occur if massive data arrays were passed back through the multiprocessing queues.

## 6. Software Development Process & Tooling

To maintain high code quality, predictability, and efficiency, the project template is bootstrapped with a modern Python software development lifecycle (SDLC) toolchain.

#### Developer Toolchain Setup

*   **uv:** Used for dependency management and virtual environment resolution, replacing legacy tools like pip/poetry.
*   **Ruff:** A Python linter and code formatter, ensuring consistent stylistic adherence across the codebase.
*   **Mypy:** For static type checking.
*   **Pytest:** The standard testing framework, heavily utilized to unit test mathematical operations within the Plant, state tracking in the Controller, and parameter validation within the component `from_config` factories.
*   **Git Commit Hooks:** Pre-commit hooks are configured to trigger Ruff formatting, syntax validation, mypy type checking and unit tests automatically to ensure code stability before any merge.

### 6.1. CLI Execution and Project Structure

The framework adheres to the `src/` layout best practice. The core simulation algorithms are kept clean and decoupled from execution. Simulations and experiments are exclusively triggered via a robust Command Line Interface (CLI):

`$ python main.py --config configs/experiment_01.yaml --export npz`

By routing all execution through a thin CLI script, users can script automated workflows without ever modifying the underlying control system logic. Data is cleanly routed to a disjointed `results/` directory, keeping the version-controlled codebase pristine.

## 7. Intended Use and Workflow

The framework is designed as a foundational library rather than a rigid, plug-and-play application. The primary intended use of this package is for engineers and researchers to **subclass all of the core components** with their specific, custom implementations and then orchestrate them through the provided simulation and experimentation pipelines.

To utilize the framework effectively, users should adhere to the following general workflow:

1.  **Subclass Components:** Create project-specific Python classes that inherit from the framework's base `Plant`, `Sensor`, `Estimator`, and `Controller` classes. Within these subclasses, implement the specialized mathematical models, filtering algorithms, and control laws required for the specific application.
2.  **Define Configuration:** Construct a YAML configuration file. This file must include a `class_path` for each component to dictate the exact subclasses to instantiate, alongside their specific parameters and sample times. This dynamic loading ensures that experimental parameters remain fully decoupled from the source code.
3.  **Execute Simulation/Experiments:** Invoke the framework's CLI, passing the configuration file. The orchestrator will automatically instantiate the custom subclasses, enforce the multi-rate timing requirements, execute the simulation loop (or a batch of experiments), and export the resulting data logs to disk for post-processing.
