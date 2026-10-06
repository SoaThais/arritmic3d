#ifndef LEGACY_HEAP_PROPAGATION_IMPL_H
#define LEGACY_HEAP_PROPAGATION_IMPL_H

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

        if (neigh->GetState(current_time) == Node::CellActivationState::ACTIVE) {

            float neigh_min_vel = neigh->parameters->cond_veloc_transversal_reduction * neigh->conduction_vel;
            float max_travel_time = distance / neigh_min_vel;
            float neigh_earliest_possible_activation = node->local_activation_time - max_travel_time;

            if (neigh->local_activation_time > neigh_earliest_possible_activation) {
                continue;
            }
        }

        CellEvent* ev_neigh = neigh->ScheduleActivation(node, direct_activation_time);

        if (ev_neigh != nullptr) {
            tissue.event_queue.InsertCellEvent(ev_neigh);
            inactive_neighs.push_back(neigh);
        }
    }

    return inactive_neighs;
}

template <typename APM, typename CVM>
void LegacyHeapPropagation<APM, CVM>::ProcessEvent(Tissue& tissue, CellEvent* ev) {

    Node* node_ = ev->cell_node;

    LOG::Warning(ev->event_time != tissue.tissue_time, "TriggerEvent(): In node ", node_->id, "Event time mismatch. Event time is ", ev->event_time, " while current time is ", tissue.tissue_time);

    if (tissue.tissue_time == node_->next_activation_time) {

        if (!node_->external_activation && node_->received_potential < node_->parameters->min_potential * node_->parameters->safety_factor) {

            LOG::Info(true, "TriggerEvent(): Safety factor acting. Node ", node_->id, " NOT activated with total potential ", node_->received_potential, " does not reach the minimum: ", node_->parameters->min_potential);

            node_->received_potential   = 0.0;
            node_->next_activation_time = MAX_TIME;
        }
        else {

            if (node_->Activate(tissue.tissue_time, &tissue)) {

                node_->next_deactivation_event->ChangeEvent(node_->next_deactivation_time);
                tissue.event_queue.InsertCellEvent(node_->next_deactivation_event);
                tissue.apd_variation += node_->apd_model.getDeltaAPD();

                vector<Node*> inactive_neighs = PropagateActivation(tissue, node_, tissue.tissue_time);

                if (!inactive_neighs.empty()) {
                    float potential_to_send = node_->received_potential / inactive_neighs.size() * node_->parameters->safety_factor;
                    for (auto neigh : inactive_neighs) {
                        neigh->received_potential += potential_to_send;
                    }
                }
            }

            node_->next_activation_time = MAX_TIME;
        }
    }

    if (tissue.tissue_time == node_->next_deactivation_time) {

        node_->received_potential       = 0.0;
        node_->external_activation      = false;
        node_->next_deactivation_time   = MAX_TIME;

        if (node_->next_activation_time < MAX_TIME) {

            node_->next_activation_event->ChangeEvent(node_->next_activation_time);
            tissue.event_queue.InsertCellEvent(node_->next_activation_event);

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

                CellEvent* ev_react = node_->ScheduleActivation(parent_node_, node_excitable_at_time);

                if (ev_react != nullptr) {
                    tissue.event_queue.InsertCellEvent(ev_react);
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