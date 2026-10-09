
#ifndef FRONT_PROPAGATION_SOLVER_H
#define FRONT_PROPAGATION_SOLVER_H

#include <cstddef>
#include <fstream>

enum class PropagationSolverType {
    LegacyHeap,
    FIM
};

template <typename Tissue>
class FrontPropagationSolver {
public:

    virtual ~FrontPropagationSolver() = default;

    // Identifica a implementação concreta.
    virtual PropagationSolverType Type() const = 0;

    // Inicializa ou reinicializa o estado da propagação.
    virtual void Initialize(Tissue& tissue) = 0;
    // virtual void Reset(Tissue& tissue) = 0;

    // Consulta a fila ou estrutura interna do solver.
    virtual float NextEventTime() const = 0;
    virtual bool HasPendingWork() const = 0;

    virtual std::size_t NextEventNodeIndex(const Tissue& tissue) const = 0;

    // Processa o próximo evento de propagação.
    virtual std::size_t AdvanceNext(Tissue& tissue) = 0;

    // Agenda a ativação de um nó por um estímulo externo.
    virtual void Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat ) = 0;

    // Persistência do estado interno do solver.
    virtual void SaveState(std::ofstream& state_file, const Tissue& tissue ) const = 0;
    virtual void LoadState(std::ifstream& state_file, Tissue& tissue ) = 0;

};

#endif