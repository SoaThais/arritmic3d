#ifndef LEGACY_HEAP_PROPAGATION_IMPL_H
#define LEGACY_HEAP_PROPAGATION_IMPL_H

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::Initialize(Tissue& tissue) {
    
    event_queue.Init(tissue.tissue_nodes, tissue.GetNumLiveNodes());

    propagation_states.assign(tissue.tissue_nodes.size(), PropagationState{});
    
    // for(auto& state : propagation_states) {
    //     state.next_activation_time = MAX_TIME;
    //     state.activation_parent = nullptr;
    //     state.activation_beat = -1;
    //     state.external_activation = false;
    // }

    BindEvents(tissue);
}

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::BindEvents(Tissue& tissue) {

    if(propagation_states.size() != tissue.tissue_nodes.size())
        throw std::runtime_error("LegacyHeapPropagation::BindEvents: node count mismatch.");
    
    for(size_t i = 0; i < propagation_states.size(); ++i) {
        propagation_states[i].next_activation_event   = event_queue.GetEvent(i, CellEventType::ACTIVATION);
        propagation_states[i].next_deactivation_event = event_queue.GetEvent(i, CellEventType::DEACTIVATION);
    }
}

template <typename APM, typename CVM>
float LegacyHeapPropagation<APM, CVM>::NextEventTime() const {

    if (event_queue.IsEmpty())
        return MAX_TIME;

    return event_queue.GetFirstCell()->event_time;
}

template <typename APM, typename CVM>
bool LegacyHeapPropagation<APM, CVM>::HasPendingWork() const {
    return !event_queue.IsEmpty();
}

template <typename APM, typename CVM>
std::size_t LegacyHeapPropagation<APM, CVM>::NextEventNodeIndex(const Tissue& tissue) const {
    
    if (event_queue.IsEmpty())
        throw std::runtime_error("LegacyHeapPropagation::NextEventNodeIndex: empty event queue.");

    CellEvent* ev = event_queue.GetFirstCell();

    const Node* begin = tissue.tissue_nodes.data();
    const Node* end = begin + tissue.tissue_nodes.size();

    if (ev->cell_node < begin || ev->cell_node >= end)
        throw std::runtime_error("LegacyHeapPropagation::NextEventNodeIndex: invalid event node.");

    return static_cast<std::size_t>(ev->cell_node - begin);
}

template <typename APM, typename CVM>
std::size_t LegacyHeapPropagation<APM, CVM>::AdvanceNext(Tissue& tissue) {
    
    if (event_queue.IsEmpty())
        return 0;

    CellEvent* ev       = event_queue.GetFirstCell();
    tissue.tissue_time  = ev->event_time;
    event_queue.ExtractFirstCell();
    ProcessEvent(tissue, ev);

    return 1;
}

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat) {
    
    if (node_index >= tissue.tissue_nodes.size())
        throw std::out_of_range("LegacyHeapPropagation::Stimulate: invalid node index.");

    Node* node = &tissue.tissue_nodes[node_index];
    ScheduleExternalActivation(tissue, node, activation_time, beat);
}

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::SaveState(std::ofstream& f, const Tissue& tissue) const {
    
    event_queue.SaveState(f, tissue.tissue_nodes);

    const size_t n_states = propagation_states.size();

    if(n_states != tissue.tissue_nodes.size())
        throw std::runtime_error("LegacyHeapPropagation::SaveState: node count mismatch.");
    
    f.write(reinterpret_cast<const char*>(&n_states), sizeof(n_states));
    
    for(const auto& state : propagation_states) {

        f.write(reinterpret_cast<const char*>(&state.next_activation_time), sizeof(state.next_activation_time));
        size_t parent_index = std::numeric_limits<size_t>::max();
        
        if(state.activation_parent != nullptr) {

            const Node* begin = tissue.tissue_nodes.data();
            const Node* end = begin + tissue.tissue_nodes.size();

            if(state.activation_parent < begin || state.activation_parent >= end)
                throw std::runtime_error("LegacyHeapPropagation::SaveState: activation parent is outside tissue.");
            
            parent_index = static_cast<size_t>(state.activation_parent - begin);
        }

        f.write(reinterpret_cast<const char*>(&parent_index), sizeof(parent_index));
        f.write(reinterpret_cast<const char*>(&state.activation_beat), sizeof(state.activation_beat));
        f.write(reinterpret_cast<const char*>(&state.external_activation), sizeof(state.external_activation));
    }
}

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::LoadState(std::ifstream& f, Tissue& tissue) {
    
    event_queue.LoadState(f, tissue.tissue_nodes);

    size_t n_states = 0;
    f.read(reinterpret_cast<char*>(&n_states), sizeof(n_states));

    if(!f || n_states != tissue.tissue_nodes.size())
        throw std::runtime_error("LegacyHeapPropagation::LoadState: checkpoint node count mismatch or truncated state.");

    propagation_states.assign(n_states, PropagationState{});

    for(size_t i = 0; i < n_states; ++i) {

        auto& state = propagation_states[i];
        size_t parent_index = std::numeric_limits<size_t>::max();

        f.read(reinterpret_cast<char*>(&state.next_activation_time), sizeof(state.next_activation_time));
        f.read(reinterpret_cast<char*>(&parent_index), sizeof(parent_index));
        f.read(reinterpret_cast<char*>(&state.activation_beat), sizeof(state.activation_beat));
        f.read(reinterpret_cast<char*>(&state.external_activation), sizeof(state.external_activation));
        
        if(!f)
            throw std::runtime_error("LegacyHeapPropagation::LoadState: truncated propagation state.");
        
        if(parent_index != std::numeric_limits<size_t>::max()) {
            if(parent_index >= tissue.tissue_nodes.size())
                throw std::runtime_error("LegacyHeapPropagation::LoadState: invalid activation-parent index.");
            state.activation_parent = &tissue.tissue_nodes[parent_index];
        } else {
            state.activation_parent = nullptr;
        }
    }

    BindEvents(tissue);
    
}

template <typename APM, typename CVM>
size_t LegacyHeapPropagation<APM, CVM>::NodeIndex(Tissue& tissue, Node* node) const {

    assert(node >= tissue.tissue_nodes.data());
    assert(node < tissue.tissue_nodes.data() + tissue.tissue_nodes.size());

    return static_cast<size_t>(node - tissue.tissue_nodes.data());
}

template <typename APM, typename CVM>
typename LegacyHeapPropagation<APM, CVM>::PropagationState&
LegacyHeapPropagation<APM, CVM>::GetState(Tissue& tissue, Node* node) {
    return propagation_states.at(NodeIndex(tissue, node));
}

template <typename APM, typename CVM>
typename LegacyHeapPropagation<APM, CVM>::CellEvent*
LegacyHeapPropagation<APM, CVM>::ScheduleActivation(Tissue& tissue, Node* node, Node* parent, float activation_time) {

    auto& state = GetState(tissue, node);

    if(parent->parameters->safety_factor < node->parameters->safety_factor) {
        return nullptr;
    }

    if(activation_time < state.next_activation_time) {

        state.next_activation_time = activation_time;
        state.activation_parent    = parent;
        state.activation_beat      = parent->beat;

        state.next_activation_event->ChangeEvent(state.next_activation_time);

        return state.next_activation_event;
    }

    return nullptr;
}

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::ScheduleExternalActivation(Tissue& tissue, Node* node, float activation_time, int beat_n) {
    
    auto& state = GetState(tissue, node);

    if(node->GetState(activation_time) == Node::CellActivationState::INACTIVE) {

        if(activation_time < state.next_activation_time) {

            state.next_activation_time = activation_time;

            state.next_activation_event->ChangeEvent(state.next_activation_time);

            state.external_activation   = true;
            state.activation_parent     = nullptr;
            state.activation_beat       = beat_n;

            node->received_potential = 1.0;

            event_queue.InsertCellEvent(state.next_activation_event);
        }
    }
}

template <typename APM, typename CVM>
vector<typename LegacyHeapPropagation<APM, CVM>::Node*>
LegacyHeapPropagation<APM, CVM>::PropagateActivation(Tissue& tissue, Node* node, float current_time) {

    vector<Node*> inactive_neighs;

    for (unsigned int i = 0; i < tissue.tissue_geometry.num_neighbours; ++i) {

        Node* neigh = tissue.NodeDisplace(node, tissue.tissue_geometry.displacement[i]);

        if (neigh == nullptr)
            continue;


        assert(neigh >= tissue.tissue_nodes.data() && neigh < tissue.tissue_nodes.data() + tissue.tissue_nodes.size());

        if (neigh->type == CELL_TYPE_VOID)
            continue;


        float distance = tissue.tissue_geometry.distance_to_neighbour[i];

        typename Node::Vector3 activation_dir = -tissue.tissue_geometry.relative_position[i];

        float direct_vel = node->ComputeDirectionalConductionVelocity(activation_dir);

        float direct_activation_time = node->local_activation_time + distance / direct_vel;

        // if (neigh->GetId() == 1150) {
        //     std::cout
        //         << "\n[DEBUG PROPAGATION TO 1150]"
        //         << "\n  parent=" << node->GetId()
        //         << "\n  parent_LAT=" << node->local_activation_time
        //         << "\n  distance=" << distance
        //         << "\n  direct_vel=" << direct_vel
        //         << "\n  activation_time=" << direct_activation_time
        //         << "\n";
        // }

        if (neigh->GetState(current_time) == Node::CellActivationState::ACTIVE) {

            float neigh_min_vel = neigh->parameters->cond_veloc_transversal_reduction * neigh->conduction_vel;
            float max_travel_time = distance / neigh_min_vel;
            float neigh_earliest_possible_activation = node->local_activation_time - max_travel_time;

            if (neigh->local_activation_time > neigh_earliest_possible_activation) {
                continue;
            }
        }

        // CellEvent* ev_neigh = neigh->ScheduleActivation(node, direct_activation_time);
        CellEvent* ev_neigh = ScheduleActivation(tissue, neigh, node, direct_activation_time);

        if (ev_neigh != nullptr) {
            event_queue.InsertCellEvent(ev_neigh);
            inactive_neighs.push_back(neigh);
        }
    }

    return inactive_neighs;
}

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::ProcessEvent(Tissue& tissue, CellEvent* ev) {

    Node* node_ = ev->cell_node;
    auto& state = GetState(tissue, node_);

    // if (node_->GetId() == 39246) {
    //     std::cout
    //         << "\n[DEBUG PROCESS 39246]"
    //         << "\n  tissue_time=" << tissue.tissue_time
    //         << "\n  event_time=" << ev->event_time
    //         << "\n  node_beat=" << node_->beat
    //         << "\n  activation_beat=" << state.activation_beat
    //         << "\n  next_activation_time=" << state.next_activation_time
    //         << "\n  LAT=" << node_->local_activation_time
    //         << "\n";
    // }

    LOG::Warning(ev->event_time != tissue.tissue_time, "TriggerEvent(): In node ", node_->id, "Event time mismatch. Event time is ", ev->event_time, " while current time is ", tissue.tissue_time);

    if (tissue.tissue_time == state.next_activation_time) {

        if (!state.external_activation && node_->received_potential < node_->parameters->min_potential * node_->parameters->safety_factor) {

            LOG::Info(true, "TriggerEvent(): Safety factor acting. Node ", node_->id, " NOT activated with total potential ", node_->received_potential, " does not reach the minimum: ", node_->parameters->min_potential);

            node_->received_potential   = 0.0;
            state.next_activation_time = MAX_TIME;
        }
        else {

            // if(node_->GetId() >= 1150 && node_->GetId() <= 1399) {
            //     std::cout
            //         << "[DEBUG ACTIVATE TARGET]"
            //         << " node=" << node_->GetId()
            //         << " time=" << tissue.tissue_time
            //         << " beat=" << state.activation_beat
            //         << std::endl;
            // }

            if (node_->Activate(tissue.tissue_time, state.activation_beat, &tissue)) {

                state.next_deactivation_event->ChangeEvent(node_->recovery_time);
                event_queue.InsertCellEvent(state.next_deactivation_event);
                tissue.apd_variation += node_->apd_model.getDeltaAPD();

                vector<Node*> inactive_neighs = PropagateActivation(tissue, node_, tissue.tissue_time);

                if (!inactive_neighs.empty()) {
                    float potential_to_send = node_->received_potential / inactive_neighs.size() * node_->parameters->safety_factor;
                    for (auto neigh : inactive_neighs) {
                        neigh->received_potential += potential_to_send;
                    }
                }
            }

            state.next_activation_time = MAX_TIME;
        }
    }

    if (tissue.tissue_time == node_->recovery_time) {

        node_->received_potential       = 0.0;
        state.external_activation       = false;
        node_->recovery_time            = MAX_TIME;

        if (state.next_activation_time < MAX_TIME) {

            state.next_activation_event->ChangeEvent(state.next_activation_time);
            event_queue.InsertCellEvent(state.next_activation_event);

        }

        if (tissue.long_apd_reactivation) {

            float node_excitable_at_time = node_->local_activation_time + 1.05 * node_->apd_model.getERP();

            Node* parent_node_ = nullptr;

            for (unsigned int i = 0; i < tissue.tissue_geometry.num_neighbours; ++i) {

                Node* neigh = tissue.NodeDisplace(node_, tissue.tissue_geometry.displacement[i]);

                if (neigh == nullptr)
                    continue;

                assert(neigh >= tissue.tissue_nodes.data() && neigh < tissue.tissue_nodes.data() + tissue.tissue_nodes.size());

                if (neigh->type != CELL_TYPE_VOID) {

                    if (neigh->GetState(node_excitable_at_time) == Node::CellActivationState::ACTIVE) {

                        float neigh_plateau_duration = neigh->local_activation_time + tissue.apd_plateau_duration * neigh->apd_model.getAPD();

                        if (node_excitable_at_time <= neigh_plateau_duration) {

                            node_->received_potential += 1.0;

                            if (parent_node_ == nullptr) {
                                parent_node_ = neigh;
                            }
                            else if (parent_node_->local_activation_time < neigh->local_activation_time) {
                                parent_node_ = neigh;
                            }
                        }
                    }
                }
            }

            if (node_->received_potential >= 3) {

                CellEvent* ev_react = ScheduleActivation(tissue, node_, parent_node_, node_excitable_at_time);

                if (ev_react != nullptr) {
                    event_queue.InsertCellEvent(ev_react);
                }
                else {
                    node_->received_potential = 0.0;
                }

            }
            else {
                node_->received_potential = 0.0;
            }
        }
    }
}

#endif