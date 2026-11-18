# Arritmic3D Refactoring History

This document describes the set of changes and refactorings applied to the Arritmic3D simulation core. The main goal of this refactoring was to decouple the mathematical propagation logic from cellular physiology, enabling the integration of new propagation solvers.

## Overview of Changes

The cardiac tissue architecture (`CardiacTissue` and `BasicTissue`) was restructured to be agnostic to the stimulus propagation method. Previously, the event queue and activation logic were tightly coupled to the node class (`NodeT`). With this refactoring:

1. Cellular physiology (`NodeT`) was isolated from the propagation data structures.
2. System events were separated from cellular events.
3. The main time loop was moved internally into C++.
4. An initial implementation of new propagation algorithms (FIM and FIM Eikonal) was developed following a standardized interface.

---

## Main Technical Changes

### 1. Separation of System and Cellular Events

- **`SystemEventScheduler`:** A new class responsible exclusively for managing system-level events.
- **`CellEventQueue`:** Now dedicated entirely to managing cell-level events (depolarization/repolarization), without mixing them with system-level logic.

### 2. Abstract Solver Interface

- **`FrontPropagationSolver`:** A purely virtual base class was introduced to define the interface for any propagation algorithm, with methods such as `Initialize`, `AdvanceUntil`, `Stimulate`, `SaveState`, and `LoadState`.
- **`PropagationSolverType`:** An enumeration exported to Python, allowing users to select the propagation solver (`LEGACY_HEAP`, `FIMEikonal`).

**Available Solvers:**

- **LegacyHeapPropagation:** Encapsulates the original Arritmic3D propagation algorithm, based on a priority queue (heap).
- **FIMEikonalPropagation:** A solver based on the solution of the Eikonal equation using the Fast Iterative Method (FIM).

---

### 3. Isolation of the Physiological State (`NodeT`)

- Variables related to the propagation solver were removed from the `NodeT` class.
- Each solver now maintains its own internal state structure, allowing `NodeT` to handle exclusively electrophysiological properties.
- The `at_history` and `apd_history` vectors were added to `NodeT` to track all cell activation times and action potential durations throughout the simulation.

## New Features and Utilities

- **Extended Output Formats:**
  - A new C++ method, `SaveActivationTimesACM`, was introduced (exposed as the `--save-acm` CLI argument) to generate text files containing the activation time and action potential duration histories for each cell.
- **Pre-scheduling of Stimuli:**
  - All stimulation protocols (BCL and number of stimuli) are processed beforehand and scheduled in C++ through `ScheduleActivation`.

---

## Validation and Testing

- **Regression Tests (Refactoring):** To ensure strict consistency between the original code and the refactored implementation using `LegacyHeapPropagation`, three scenarios were validated:
  - Isotropic stimulation
  - Anisotropic stimulation
  - S1-S2 stimulation protocol

  The corresponding test configurations are available in `./Tests/Refactor_Cases/`.

- **Solver Comparison (Legacy vs. FIM):** To analyze differences in propagation dynamics between the legacy solver and the new FIM-based implementations, the following scenarios were compared:
  - Wavefront propagation with a circular stimulus
  - Wavefront propagation with a concave stimulus
  - Planar stimulation with varying fiber orientations (sweep from 0° to 90°)

  The corresponding test configurations are available in `./Tests/FIM_Cases/`.