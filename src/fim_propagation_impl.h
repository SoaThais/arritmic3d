#ifndef FIM_PROPAGATION_IMPL_H
#define FIM_PROPAGATION_IMPL_H

#include <cmath>
#include <limits>
#include <utility>

// ---------------------------------------------------------------------------
// Inicialização, suporte e consulta
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::Initialize (Tissue& tissue) {    
    states.assign(tissue.tissue_nodes.size(), PropagationState{});
    list_.clear();
    converged_.clear();
    recovery_queue_.clear();
}

template <typename APM, typename CVM>
std::size_t FIMPropagation<APM, CVM>::NodeIndex(const Tissue& tissue, const Node* node) const {
    assert(node >= tissue.tissue_nodes.data());
    assert(node < tissue.tissue_nodes.data() + tissue.tissue_nodes.size());
    return static_cast<std::size_t>(node - tissue.tissue_nodes.data());
}

template <typename APM, typename CVM>
bool FIMPropagation<APM, CVM>::HasPendingWork() const {

    for (std::size_t j : list_)
        if (states[j].status == Status::LIST)
            return true;

    for (std::size_t j : converged_)
        if (states[j].status == Status::CONVERGED)
            return true;

    // Recuperações pendentes: o solver legado mantém eventos de desativação
    // na fila enquanto houver células ativas.
    return !recovery_queue_.empty();
}

// ---------------------------------------------------------------------------
// Sondas: consultam o estado fisiológico do nó sem modificá-lo
// ---------------------------------------------------------------------------

/**
 * Equivale a: o nó está inativo em 'at' e Node::Activate(at) seria aceito.
 */
template <typename APM, typename CVM>
bool FIMPropagation<APM, CVM>::CanActivateAt(const Node& node, float at) const {

    if (node.type == CELL_TYPE_VOID)
        return false;

    // Candidato anterior à última ativação (fonte obsoleta).
    if (at < node.apd_model.getActivationTime())
        return false;

    if (node.apd_model.IsActive(at))
        return false;

    APM probe = node.apd_model;
    return probe.Activate(at);
}

/**
 * Velocidade de condução longitudinal que o nó teria se fosse ativado em 'at'.
 * Replica NodeT::ComputeActivation sobre uma cópia do cv_model.
 */
template <typename APM, typename CVM>
float FIMPropagation<APM, CVM>::ProbeConductionVelocity(const Node& node, float at) const {

    const float prev_di  = node.apd_model.getDI(at);
    const float prev_apd = node.apd_model.getAPD();

    CVM cv = node.cv_model;
    cv.Activate(prev_di, prev_apd);

    return cv.getConductionVelocity();
}

/**
 * Velocidade direcional do nó para uma velocidade longitudinal dada
 * (a fórmula de NodeT::ComputeDirectionalConductionVelocity é linear em
 * conduction_vel; a troca é temporária).
 */
template <typename APM, typename CVM>
float FIMPropagation<APM, CVM>::DirectionalVelocity(Node& node, float cond_vel, const typename Node::Vector3& dir) const {

    const float saved = node.conduction_vel;

    node.conduction_vel = cond_vel;
    const float v = node.ComputeDirectionalConductionVelocity(dir);
    node.conduction_vel = saved;

    return v;
}

// ---------------------------------------------------------------------------
// Lista ativa
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::Enqueue(std::size_t j) {

    if (!states[j].queued) {
        states[j].queued = true;
        list_.push_back(j);
    }
}

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::SetTentative(Tissue& tissue, std::size_t j, float at, std::int64_t parent, int beat, bool external, double potential) {

    auto& s = states[j];

    s.tentative_at        = at;
    s.velocity            = ProbeConductionVelocity(tissue.tissue_nodes[j], at);
    s.parent              = parent;
    s.beat                = beat;
    s.external            = external;
    s.status              = Status::LIST;
    s.tentative_potential = potential;
    s.source_at           = MAX_TIME;

    Enqueue(j);
}

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::ClearTentative(std::size_t j) {

    const PropagationState old = states[j];

    states[j]             = PropagationState{};
    states[j].queued      = old.queued;
    states[j].source_at   = old.source_at;    // a fonte sobrevive à efetivação
    states[j].source_vel  = old.source_vel;
    states[j].source_beat = old.source_beat;
}

template <typename APM, typename CVM>
float FIMPropagation<APM, CVM>::MinListTime() const {

    float t_min = MAX_TIME;

    for (std::size_t j : list_) {
        const auto& s = states[j];
        if (s.status == Status::LIST && s.tentative_at < t_min)
            t_min = s.tentative_at;
    }

    return t_min;
}

// ---------------------------------------------------------------------------
// Estímulo externo
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat) {

    if (node_index >= tissue.tissue_nodes.size())
        throw std::out_of_range("FIMPropagation::Stimulate: invalid node index.");

    Node& node = tissue.tissue_nodes[node_index];
    auto& s    = states[node_index];

    if (node.GetState(activation_time) != Node::CellActivationState::INACTIVE)
        return;

    if (!(activation_time < s.tentative_at))
        return;

    // O legado agenda e depois falha em Activate(); aqui a falha é antecipada
    // para que filhos nunca derivem de uma ativação que não vai ocorrer.
    if (!CanActivateAt(node, activation_time))
        return;

    SetTentative(tissue, node_index, activation_time, -1, beat, true, 1.0f);

    node.received_potential = 1.0;
}

// ---------------------------------------------------------------------------
// FIM: relaxação por janela
// ---------------------------------------------------------------------------

/**
 * Melhor candidato a tempo de ativação do nó j, puxando dos vizinhos que já
 * são fontes estáveis (convergidos ou já efetivados; nunca nós ainda na
 * lista): T_j = T_i + d_ij / c_ij, com c_ij a velocidade direcional do nó i.
 * Só aceita candidatos em que j pode de fato ser ativado (CanActivateAt).
 */
// template <typename APM, typename CVM>
// bool FIMPropagation<APM, CVM>::BestCandidate(Tissue& tissue, std::size_t j, Candidate& best) {

//     Node& node       = tissue.tissue_nodes[j];
//     const auto& geom = tissue.tissue_geometry;

//     bool found = false;
//     best.at = MAX_TIME;

//     for (unsigned int k = 0; k < geom.num_neighbours; ++k) {

//         Node* src = tissue.NodeDisplace(&node, geom.displacement[k]);

//         if (src == nullptr || src->type == CELL_TYPE_VOID)
//             continue;

//         const std::size_t i = NodeIndex(tissue, src);
//         const auto& si      = states[i];

//         // Fonte precisa ter valor estável: nós ainda na lista não propagam.
//         if (si.status == Status::LIST || si.source_at == MAX_TIME)
//             continue;

//         // Mesma regra de ScheduleActivation do legado.
//         if (src->parameters->safety_factor < node.parameters->safety_factor)
//             continue;

//         const typename Node::Vector3 activation_dir = - geom.relative_position[k];

//         const float v = DirectionalVelocity(*src, si.source_vel, activation_dir);

//         if (!(v > 0.0f))
//             continue;

//         const float distance = geom.distance_to_neighbour[k];

//         // Reproduz a condição do legado para vizinhos ativos
//         // no instante de ativação da fonte.
//         if (node.apd_model.IsActive(si.source_at)) {

//             const float reduction = node.parameters->cond_veloc_transversal_reduction;

//             const float min_vel = reduction * node.conduction_vel;

//             if (min_vel > 0.0f) {

//                 const float max_travel_time = distance / min_vel;

//                 const float earliest_possible_activation =
//                     si.source_at - max_travel_time;

//                 if (node.apd_model.getActivationTime() >
//                     earliest_possible_activation) {
//                     continue;
//                 }
//             }
//         }

//         const float at = si.source_at + distance / v;

//         // Candidato no passado: fonte obsoleta (onda de batimento anterior).
//         if (at < tissue.tissue_time)
//             continue;

//         if (at >= best.at)
//             continue;

//         if (!CanActivateAt(node, at))
//             continue;

//         best.at     = at;
//         best.parent = static_cast<std::int64_t>(i);
//         best.beat   = si.source_beat;
//         found       = true;
//     }

//     return found;
// }

template <typename APM, typename CVM>
bool FIMPropagation<APM, CVM>::BestCandidate(Tissue& tissue, std::size_t j, Candidate& best) {

    Node& node       = tissue.tissue_nodes[j];
    const auto& geom = tissue.tissue_geometry;

    struct IncomingCharge {
        float arrival_time;
        float potential;
        std::int64_t parent;
        int beat;
    };

    std::vector<IncomingCharge> charges;
    charges.reserve(geom.num_neighbours);

    for (unsigned int k = 0; k < geom.num_neighbours; ++k) {

        Node* src = tissue.NodeDisplace(&node, geom.displacement[k]);

        if (src == nullptr || src->type == CELL_TYPE_VOID) continue;

        const std::size_t i = NodeIndex(tissue, src);
        const auto& si      = states[i];

        if (si.status == Status::LIST || si.source_at == MAX_TIME) continue;
        if (src->parameters->safety_factor < node.parameters->safety_factor) continue;

        const typename Node::Vector3 activation_dir = - geom.relative_position[k];
        const float v = DirectionalVelocity(*src, si.source_vel, activation_dir);

        if (!(v > 0.0f)) continue;

        const float distance = geom.distance_to_neighbour[k];

        if (node.apd_model.IsActive(si.source_at)) {

            const float reduction = node.parameters->cond_veloc_transversal_reduction;
            const float min_vel   = reduction * node.conduction_vel;

            if (min_vel > 0.0f) {
                const float earliest_possible_activation = si.source_at - (distance / min_vel);
                if (node.apd_model.getActivationTime() > earliest_possible_activation) continue;
            }
        }

        const float at = si.source_at + distance / v;

        // // --- DEBUG FIM ---
        // if (node.id == 125132) {
        //     printf("[FIM] Alvo: 125132 | Fonte: %d | T_fonte: %.8f | Dist: %.8f | Vel: %.8f | TA_calc: %.8f\n",
        //            src->id, si.source_at, distance, v, at);
        // }
        // // -----------------

        if (at < tissue.tissue_time) continue;

        // Ao invés de checar o CanActivateAt aqui, nós guardamos a carga recebida
        charges.push_back({at, si.outgoing_potential, static_cast<std::int64_t>(i), si.source_beat});
    }

    if (charges.empty()) return false;

    // Ordena do tempo de chegada mais rápido para o mais lento
    std::sort(charges.begin(), charges.end(), [](const IncomingCharge& a, const IncomingCharge& b) {
        return a.arrival_time < b.arrival_time;
    });

    float accumulated = 0.0f;
    float threshold = node.parameters->min_potential * node.parameters->safety_factor;

    // Vai somando no tempo até bater o limiar!
    for (const auto& charge : charges) {

        accumulated += charge.potential;
        
        if (accumulated >= threshold) {
            // Só aceita se a célula puder ativar neste tempo específico
            if (CanActivateAt(node, charge.arrival_time)) {
                best.at        = charge.arrival_time;
                best.parent    = charge.parent;
                best.beat      = charge.beat;
                best.potential = accumulated;
                return true; 
            }
            // Se estava refratária, o laço continua para ver se tempos mais 
            // tardios conseguem ativar (quando ela sair do período refratário).
        }
    }

    return false;
}

/**
 * Equivalente ao active_neighbors do FIM original: o nó i acabou de
 * convergir; vizinhos fora da lista (em repouso ou já convergidos) são
 * reavaliados e, se melhorarem, entram (ou voltam) para a lista.
 * Vizinhos que já estão na lista se recalculam na própria varredura.
 */
template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::ActivateNeighbors(Tissue& tissue, std::size_t i) {

    Node& node       = tissue.tissue_nodes[i];
    const auto& geom = tissue.tissue_geometry;

    for (unsigned int k = 0; k < geom.num_neighbours; ++k) {

        Node* neigh = tissue.NodeDisplace(&node, geom.displacement[k]);

        if (neigh == nullptr || neigh->type == CELL_TYPE_VOID)
            continue;

        const std::size_t j = NodeIndex(tissue, neigh);
        auto& sj            = states[j];

        if (sj.status == Status::LIST)
            continue;

        Candidate c;

        if (!BestCandidate(tissue, j, c))
            continue;

        if (sj.status == Status::REST || c.at < sj.tentative_at - params.epsilon)
            SetTentative(tissue, j, c.at, c.parent, c.beat, false, c.potential);
    }
}

/**
 * Relaxação por janela [t_start, t_start + window]. Em cada varredura, cada
 * nó da lista dentro da janela recalcula seu AT puxando dos vizinhos
 * (BestCandidate) e compara com o valor anterior:
 *  - se melhorou, atualiza e continua na lista;
 *  - se não mudou, converge: sai da lista, passa a valer como fonte e
 *    reavalia os vizinhos (ActivateNeighbors).
 * Repete até não restar nó da lista dentro da janela.
 */
template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::RelaxWindow(Tissue& tissue, float t_start) {

    const float window_end = t_start + params.window;
    std::size_t sweeps     = 0;

    while (true) {

        bool changed = false;

        vector<std::size_t> current;
        current.swap(list_);

        for (std::size_t j : current) {

            auto& s  = states[j];
            s.queued = false;

            if (s.status != Status::LIST)
                continue;   // entrada obsoleta

            // Todo nó da lista recalcula o AT a cada varredura; a janela só
            // decide quem pode convergir.
            Candidate c;

            bool bestCandidate = BestCandidate(tissue, j, c);

            const bool improved = bestCandidate && c.at < s.tentative_at - params.epsilon;

            if (improved) {
                SetTentative(tissue, j, c.at, c.parent, c.beat, false, c.potential);   // continua na lista
                changed = true;
                continue;
            }

            if (s.tentative_at > window_end) {
                Enqueue(j); // fora da janela: continua na lista sem convergir
                continue;
            }

            // if (tissue.tissue_nodes[j].id == 170)
            // printf("ID %d BestCandidate %d External %d AT New %f AT Old %f Convergiu %d\n", tissue.tissue_nodes[j].id, bestCandidate, s.external, c.at, s.tentative_at, (bestCandidate && fabs(c.at - s.tentative_at) < params.epsilon) || s.external);

            // Convergiu
            s.status       = Status::CONVERGED;
            s.source_at    = s.tentative_at;
            s.source_vel   = s.velocity;
            s.source_beat  = s.beat;
            s.source_potential = s.tentative_potential;
            converged_.push_back(j);

            // --- CONTAGEM DE SUMIDOUROS ---

            int sinks = 0;
            const auto& geom = tissue.tissue_geometry;

            for (unsigned int k = 0; k < geom.num_neighbours; ++k) {

                Node* neigh = tissue.NodeDisplace(&tissue.tissue_nodes[j], geom.displacement[k]);
                
                if (neigh == nullptr || neigh->type == CELL_TYPE_VOID) continue;
                
                std::size_t n_idx = NodeIndex(tissue, neigh);
                const auto& sn = states[n_idx];
                
                // Se o vizinho ainda não ativou, ou vai ativar depois de mim, ele é um sumidouro
                if (sn.source_at == MAX_TIME || sn.source_at > s.source_at) {
                    sinks++;
                }

            }

            // Fração de carga que será puxada pelo BestCandidate dos vizinhos
            s.outgoing_potential = (s.source_potential / std::max(1, sinks)) * tissue.tissue_nodes[j].parameters->safety_factor;

            ActivateNeighbors(tissue, j);
            changed = true;
        }

        if (!changed)
            break;

        if (++sweeps > params.max_sweeps)
            throw std::runtime_error("FIMPropagation::RelaxWindow: max_sweeps exceeded.");
    }
}

// ---------------------------------------------------------------------------
// Efetivação
// ---------------------------------------------------------------------------

/**
 * Efetiva, em ordem de tempo, os nós convergidos cujo valor é definitivo
 * (<= menor valor da lista - tau_s) e que estão dentro do limite pedido.
 */
template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::CommitFinalized(Tissue& tissue, float t_list, float limit, bool inclusive) {

    const float safe = (t_list == MAX_TIME) ? MAX_TIME : t_list - params.tau_s;

    vector<std::pair<float, std::size_t>> batch;
    vector<std::size_t> keep;

    for (std::size_t j : converged_) {

        const auto& s = states[j];

        if (s.status != Status::CONVERGED)
            continue;   // obsoleta (reentrou na lista ou já efetivada)

        const float at = s.tentative_at;

        const bool is_final = (safe == MAX_TIME) || (at <= safe);
        const bool in_limit = inclusive ? (at <= limit) : (at < limit);

        if (is_final && in_limit)
            batch.emplace_back(at, j);
        else
            keep.push_back(j);
    }

    converged_.swap(keep);

    std::sort(batch.begin(), batch.end());

    for (const auto& item : batch) {

        // Pode ter sido efetivado antes (entrada duplicada).
        if (states[item.second].status != Status::CONVERGED)
            continue;

        CommitNode(tissue, item.second);
    }
}

// template <typename APM, typename CVM>
// void FIMPropagation<APM, CVM>::CommitNode(Tissue& tissue, std::size_t idx) {

//     Node& node = tissue.tissue_nodes[idx];

//     const float at   = states[idx].tentative_at;
//     const int   beat = states[idx].beat;

//     ClearTentative(idx);

//     // Recuperação anterior pendente (mantém a ordem dos dados do sensor).
//     if (node.recovery_time != MAX_TIME && node.recovery_time <= at)
//         ApplyRecovery(tissue, idx);

//     tissue.tissue_time = at;

//     if (node.Activate(at, beat, &tissue)) {

//         tissue.apd_variation += node.apd_model.getDeltaAPD();
//         recovery_queue_.push_back(idx);

//         // Potencial enviado aos filhos (informativo, como no legado).
//         const auto& geom = tissue.tissue_geometry;
//         unsigned int n_children = 0;

//         for (unsigned int k = 0; k < geom.num_neighbours; ++k) {
//             Node* neigh = tissue.NodeDisplace(&node, geom.displacement[k]);
//             if (neigh == nullptr || neigh->type == CELL_TYPE_VOID)
//                 continue;
//             const auto& sn = states[NodeIndex(tissue, neigh)];
//             if (sn.status != Status::REST && sn.parent == static_cast<std::int64_t>(idx))
//                 ++n_children;
//         }

//         if (n_children > 0) {
//             const float potential_to_send = node.received_potential / n_children * node.parameters->safety_factor;
//             for (unsigned int k = 0; k < geom.num_neighbours; ++k) {
//                 Node* neigh = tissue.NodeDisplace(&node, geom.displacement[k]);
//                 if (neigh == nullptr || neigh->type == CELL_TYPE_VOID)
//                     continue;
//                 const auto& sn = states[NodeIndex(tissue, neigh)];
//                 if (sn.status != Status::REST && sn.parent == static_cast<std::int64_t>(idx))
//                     neigh->received_potential += potential_to_send;
//             }
//         }
//     }

//     if (node.parameters != nullptr && node.parameters->sensor)
//         tissue.sensor_dict.AddData(node.id, node.GetData(at));
// }

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::CommitNode(Tissue& tissue, std::size_t idx) {

    Node& node = tissue.tissue_nodes[idx];

    const float at   = states[idx].tentative_at;
    const int   beat = states[idx].beat;
    
    const float final_potential = states[idx].source_potential; 

    ClearTentative(idx);

    if (node.recovery_time != MAX_TIME && node.recovery_time <= at)
        ApplyRecovery(tissue, idx);

    tissue.tissue_time = at;
    
    node.received_potential = final_potential;

    if (node.Activate(at, beat, &tissue)) {
        tissue.apd_variation += node.apd_model.getDeltaAPD();
        recovery_queue_.push_back(idx);
    }

    if (node.parameters != nullptr && node.parameters->sensor)
        tissue.sensor_dict.AddData(node.id, node.GetData(at));
}

// ---------------------------------------------------------------------------
// Recuperação
// ---------------------------------------------------------------------------

// template <typename APM, typename CVM>
// void FIMPropagation<APM, CVM>::ApplyRecovery(Tissue& tissue, std::size_t idx) {

//     Node& node = tissue.tissue_nodes[idx];

//     if (node.recovery_time == MAX_TIME)
//         return;

//     const float recovery_time = node.recovery_time;

//     node.received_potential = 0.0;
//     node.recovery_time      = MAX_TIME;

//     if (node.parameters != nullptr && node.parameters->sensor)
//         tissue.sensor_dict.AddData(node.id, node.GetData(recovery_time));
// }

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::ApplyRecovery(Tissue& tissue, std::size_t idx) {

    Node& node = tissue.tissue_nodes[idx];

    if (node.recovery_time == MAX_TIME)
        return;

    const float recovery_time = node.recovery_time;

    node.received_potential = 0.0;
    node.recovery_time      = MAX_TIME;

    if (node.parameters != nullptr && node.parameters->sensor)
        tissue.sensor_dict.AddData(node.id, node.GetData(recovery_time));

    // --- REATIVAÇÃO POR APD LONGO ---
    if (tissue.long_apd_reactivation) {
        
        // 5% de margem no ERP para a janela de vulnerabilidade
        float node_excitable_at_time = node.local_activation_time + 1.05f * node.apd_model.getERP();

        const auto& geom = tissue.tissue_geometry;
        Node* parent_node_ = nullptr;
        float reactivation_potential = 0.0f;

        for (unsigned int k = 0; k < geom.num_neighbours; ++k) {
            Node* neigh = tissue.NodeDisplace(&node, geom.displacement[k]);

            if (neigh == nullptr || neigh->type == CELL_TYPE_VOID)
                continue;

            // Checa se o vizinho estará ativo no momento em que esta célula ficar vulnerável
            if (neigh->GetState(node_excitable_at_time) == Node::CellActivationState::ACTIVE) {
                
                float neigh_plateau_duration = neigh->local_activation_time + tissue.apd_plateau_duration * neigh->apd_model.getAPD();

                if (node_excitable_at_time <= neigh_plateau_duration) {
                    
                    reactivation_potential += 1.0f;

                    // O "pai" da nova onda será o vizinho ativo mais recente
                    if (parent_node_ == nullptr) {
                        parent_node_ = neigh;
                    }
                    else if (parent_node_->local_activation_time < neigh->local_activation_time) {
                        parent_node_ = neigh;
                    }
                }
            }
        }

        // Limiar fixo de 3 vizinhos em platô herdado do Arritmic3D
        if (reactivation_potential >= 3.0f) {
            
            // O legado barra a reativação se o pai tiver safety_factor menor
            if (parent_node_->parameters->safety_factor >= node.parameters->safety_factor) {
                
                std::int64_t parent_idx = NodeIndex(tissue, parent_node_);
                int parent_beat = states[parent_idx].source_beat;

                // Injeta a reativação na malha.
                SetTentative(tissue, idx, node_excitable_at_time, parent_idx, parent_beat, true, reactivation_potential);
            }
        }
    }
}

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::ProcessRecoveries(Tissue& tissue, float limit, bool inclusive) {

    if (recovery_queue_.empty())
        return;

    vector<std::pair<float, std::size_t>> due;
    vector<std::size_t> keep;

    for (std::size_t idx : recovery_queue_) {

        const float rt = tissue.tissue_nodes[idx].recovery_time;

        if (rt == MAX_TIME)
            continue;   // já processada

        const bool is_due = inclusive ? (rt <= limit) : (rt < limit);

        if (is_due)
            due.emplace_back(rt, idx);
        else
            keep.push_back(idx);
    }

    std::sort(due.begin(), due.end());

    float last_recovery_time = -1.0f; // <-- ADICIONADO

    for (const auto& item : due) {
        ApplyRecovery(tissue, item.second);
        last_recovery_time = item.first; // O tempo de recuperação vem de 'due'
    }

    // --- CORREÇÃO DO RELÓGIO GLOBAL ---
    // O Legacy avança o tissue_time no final de cada evento de desativação
    if (last_recovery_time > 0.0f && last_recovery_time > tissue.tissue_time) {
        tissue.tissue_time = last_recovery_time;
    }
    // ----------------------------------

    recovery_queue_.swap(keep);
}

// ---------------------------------------------------------------------------
// AdvanceUntil
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::AdvanceUntil(Tissue& tissue, float time_limit, bool inclusive) {

    if (time_limit < tissue.GetTime())
        throw std::runtime_error("Cannot advance propagation backwards.");

    auto within_limit = [&](float t) {
        return inclusive ? (t <= time_limit) : (t < time_limit);
    };

    while (true) {

        float t_list = MinListTime();

        // Só relaxa janelas que começam dentro do limite: o solver nunca
        // adianta mais de uma janela além do que foi pedido.
        const bool relax = (t_list != MAX_TIME) && within_limit(t_list);

        if (relax) {
            RelaxWindow(tissue, t_list);
            t_list = MinListTime();
        }

        CommitFinalized(tissue, t_list, time_limit, inclusive);

        if (!relax)
            break;
    }

    ProcessRecoveries(tissue, time_limit, inclusive);
}

// ---------------------------------------------------------------------------
// Persistência
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::SaveState(std::ofstream& f, const Tissue& tissue) const {

    constexpr int solver_id     = static_cast<int>(PropagationSolverType::FIM);
    constexpr int state_version = 2;

    f.write(reinterpret_cast<const char*>(&solver_id), sizeof(solver_id));
    f.write(reinterpret_cast<const char*>(&state_version), sizeof(state_version));

    if (!f)
        throw std::runtime_error("FIMPropagation::SaveState: failed to write header.");

    const std::size_t n_states = states.size();

    if (n_states != tissue.tissue_nodes.size())
        throw std::runtime_error("FIMPropagation::SaveState: node count mismatch.");

    f.write(reinterpret_cast<const char*>(&n_states), sizeof(n_states));

    for (const auto& s : states) {

        const unsigned char status = static_cast<unsigned char>(s.status);

        f.write(reinterpret_cast<const char*>(&s.tentative_at), sizeof(s.tentative_at));
        f.write(reinterpret_cast<const char*>(&s.velocity), sizeof(s.velocity));
        f.write(reinterpret_cast<const char*>(&s.parent), sizeof(s.parent));
        f.write(reinterpret_cast<const char*>(&s.beat), sizeof(s.beat));
        f.write(reinterpret_cast<const char*>(&s.external), sizeof(s.external));
        f.write(reinterpret_cast<const char*>(&status), sizeof(status));
        f.write(reinterpret_cast<const char*>(&s.source_at), sizeof(s.source_at));
        f.write(reinterpret_cast<const char*>(&s.source_vel), sizeof(s.source_vel));
        f.write(reinterpret_cast<const char*>(&s.source_beat), sizeof(s.source_beat));
    }

    if (!f)
        throw std::runtime_error("FIMPropagation::SaveState: failed to write propagation state.");
}

template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::LoadState(std::ifstream& f, Tissue& tissue) {

    int solver_id     = -1;
    int state_version = -1;

    f.read(reinterpret_cast<char*>(&solver_id), sizeof(solver_id));
    f.read(reinterpret_cast<char*>(&state_version), sizeof(state_version));

    const int expected_id = static_cast<int>(PropagationSolverType::FIM);

    if (!f || solver_id != expected_id)
        throw std::runtime_error("FIMPropagation::LoadState: solver type mismatch.");

    if (state_version != 2)
        throw std::runtime_error("FIMPropagation::LoadState: unsupported state version.");

    std::size_t n_states = 0;
    f.read(reinterpret_cast<char*>(&n_states), sizeof(n_states));

    if (!f || n_states != tissue.tissue_nodes.size())
        throw std::runtime_error("FIMPropagation::LoadState: checkpoint node count mismatch or truncated state.");

    states.assign(n_states, PropagationState{});

    for (std::size_t i = 0; i < n_states; ++i) {

        auto& s = states[i];
        unsigned char status = 0;

        f.read(reinterpret_cast<char*>(&s.tentative_at), sizeof(s.tentative_at));
        f.read(reinterpret_cast<char*>(&s.velocity), sizeof(s.velocity));
        f.read(reinterpret_cast<char*>(&s.parent), sizeof(s.parent));
        f.read(reinterpret_cast<char*>(&s.beat), sizeof(s.beat));
        f.read(reinterpret_cast<char*>(&s.external), sizeof(s.external));
        f.read(reinterpret_cast<char*>(&status), sizeof(status));
        f.read(reinterpret_cast<char*>(&s.source_at), sizeof(s.source_at));
        f.read(reinterpret_cast<char*>(&s.source_vel), sizeof(s.source_vel));
        f.read(reinterpret_cast<char*>(&s.source_beat), sizeof(s.source_beat));

        if (!f)
            throw std::runtime_error("FIMPropagation::LoadState: truncated propagation state.");

        if (status > static_cast<unsigned char>(Status::CONVERGED))
            throw std::runtime_error("FIMPropagation::LoadState: invalid node status.");

        if (s.parent >= static_cast<std::int64_t>(n_states))
            throw std::runtime_error("FIMPropagation::LoadState: invalid parent index.");

        s.status = static_cast<Status>(status);
        s.queued = false;
    }

    RebuildQueues(tissue);
}

/**
 * Reconstrói as filas internas a partir do estado restaurado.
 */
template <typename APM, typename CVM>
void FIMPropagation<APM, CVM>::RebuildQueues(Tissue& tissue) {

    list_.clear();
    converged_.clear();
    recovery_queue_.clear();

    for (std::size_t i = 0; i < states.size(); ++i) {

        if (states[i].status == Status::LIST)
            Enqueue(i);
        else if (states[i].status == Status::CONVERGED)
            converged_.push_back(i);

        if (tissue.tissue_nodes[i].recovery_time != MAX_TIME)
            recovery_queue_.push_back(i);
    }
}

#endif // FIM_PROPAGATION_IMPL_H