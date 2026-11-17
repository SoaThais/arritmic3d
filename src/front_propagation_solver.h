#ifndef FRONT_PROPAGATION_SOLVER_H
#define FRONT_PROPAGATION_SOLVER_H

#include <cstddef>
#include <fstream>

enum class PropagationSolverType {
    LegacyHeap,
    FIMEikonal
};

template <typename Tissue>
class FrontPropagationSolver {
public:

    virtual ~FrontPropagationSolver() = default;

    // Identifies the concrete implementation.
    virtual PropagationSolverType Type() const = 0;

    // Initializes or reinitializes the propagation state.
    virtual void Initialize(Tissue& tissue) = 0;
    // virtual void Reset(Tissue& tissue) = 0;

    // Queries the solver's internal queue or structure.
    virtual bool HasPendingWork() const = 0;

    // Advances internal propagation up to the time limit.
    //
    // inclusive = false: processes instants < time_limit
    // inclusive = true:  processes instants <= time_limit
    //
    // No physiological activation or recovery after
    // the requested limit may be committed.
    virtual void AdvanceUntil(Tissue& tissue, float time_limit, bool inclusive) = 0;

    // Schedules the activation of a node by an external stimulus.
    virtual void Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat ) = 0;

    // Persistence of the solver's internal state.
    virtual void SaveState(std::ofstream& state_file, const Tissue& tissue ) const = 0;
    virtual void LoadState(std::ifstream& state_file, Tissue& tissue ) = 0;

};

#endif