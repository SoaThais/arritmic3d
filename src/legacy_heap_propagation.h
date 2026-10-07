#ifndef LEGACY_HEAP_PROPAGATION_H
#define LEGACY_HEAP_PROPAGATION_H

#include <vector>
#include <cassert>

#include "geometry.h"
#include "cell_event_queue.h"

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
class LegacyHeapPropagation {

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

        void Initialize(Tissue& tissue);

        /**
        * @brief Processa um evento celular usando o algoritmo legado.
        *
        * Este método é, propositalmente, uma cópia estrutural do antigo
        * CardiacTissue::TriggerEvent().
        */
        void ProcessEvent(Tissue& tissue, CellEvent* ev);

        CellEvent* ScheduleExternalActivation(Tissue& tissue, Node* node, float activation_time, int beat_n);

    private:

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