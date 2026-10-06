#ifndef FRONT_PROPAGATION_SOLVER_H
#define FRONT_PROPAGATION_SOLVER_H

#include <cstddef>
#include <fstream>

template <typename Node>
class FrontPropagationSolver {

    public:

        virtual ~FrontPropagationSolver() = default;
        virtual void Attach(Node* nodes, size_t n_nodes) = 0;
        virtual void Reset() = 0;
        virtual void Stimulate(Node* node, float activation_time) = 0;
        virtual void AdvanceUntil(float time) = 0;
        virtual bool HasPendingWork() const = 0;
        virtual void SaveState(std::ofstream& state_file) const = 0;
        virtual void LoadState(std::ifstream& state_file) = 0;
        
};

#endif