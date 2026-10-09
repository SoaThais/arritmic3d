#ifndef LEGACY_HEAP_PROPAGATION_H
#define LEGACY_HEAP_PROPAGATION_H

#include <vector>
#include <cassert>
#include <cstddef>
#include <fstream>

#include "geometry.h"
#include "cell_event_queue.h"
#include "front_propagation_solver.h"

using std::vector;

/**
 * @brief Encapsula a implementação atual da propagação.
 *
 * Esta classe preserva o algoritmo legado de propagação do Arritmic3D.
 *
 * O objetivo deste estágio é somente retirar a implementação de
 * CardiacTissue::TriggerEvent(), sem alterar o comportamento numérico.
 *
 * A implementação continua utilizando:
 *
 *  - CellEventQueue;
 *  - tempos de ativação;
 *  - activation parent;
 *  - propagação de beat;
 *  - cálculo de velocidade direcional;
 *  - safety factor;
 *  - eventos de desativação;
 *  - long-APD reactivation.
 *
 * Posteriormente esta classe poderá ser substituída por uma implementação
 * baseada em FIM.
 */
template <typename APM, typename CVM>
class LegacyHeapPropagation : public FrontPropagationSolver<CardiacTissue<APM, CVM>> {

    public:

        using Tissue = CardiacTissue<APM, CVM>;
        using Node = NodeT<APM, CVM>;
        using CellEvent = Event<Node>;

        struct PropagationState {
            float next_activation_time = MAX_TIME;
            Node* activation_parent = nullptr;
            int activation_beat = -1;
            bool external_activation = false;
            CellEvent* next_activation_event = nullptr;
            CellEvent* next_deactivation_event = nullptr;
        };

        LegacyHeapPropagation() = default;

        PropagationSolverType Type() const override {
            return PropagationSolverType::LegacyHeap;
        }

        void Initialize(Tissue& tissue) override;
        // void Reset(Tissue& tissue) override;

        float NextEventTime() const override;
        bool HasPendingWork() const override;

        std::size_t NextEventNodeIndex(const Tissue& tissue) const override;

        std::size_t AdvanceNext(Tissue& tissue) override;

        void Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat) override;

        // Reconnect event pointers without resetting restored propagation bookkeeping.
        void BindEvents(Tissue& tissue);
        void SaveState(std::ofstream& f, const Tissue& tissue) const override;
        void LoadState(std::ifstream& f, Tissue& tissue) override;

        /**
        * @brief Processa um evento celular usando o algoritmo legado.
        *
        * Este método é, propositalmente, uma cópia estrutural do antigo
        * CardiacTissue::TriggerEvent().
        */
        void ProcessEvent(Tissue& tissue, CellEvent* ev);

        void ScheduleExternalActivation(Tissue& tissue, Node* node, float activation_time, int beat_n);

    private:

        CellEventQueue<Node>     event_queue;
        vector<PropagationState> propagation_states;

        size_t NodeIndex(Tissue& tissue, Node* node) const;
    
        PropagationState& GetState(Tissue& tissue, Node* node);

        CellEvent* ScheduleActivation(Tissue& tissue, Node* node, Node* parent, float activation_time);

        /**
        * @brief Propaga a ativação para os vizinhos.
        *
        * Mantém a regra:
        *
        *      T_j = T_i + d_ij / c_ij
        */
        vector<Node*> PropagateActivation(Tissue& tissue, Node* node, float current_time); 

};

#include "legacy_heap_propagation_impl.h"

#endif // LEGACY_HEAP_PROPAGATION_H