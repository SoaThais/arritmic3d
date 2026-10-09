/**
 * ARRITMIC3D
 *
 * (C) CoMMLab-UV 2023
 * */

#ifndef TISSUE_H
#define TISSUE_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include <memory>

#include "geometry.h"
#include "node.h"
#include "error.h"
#include "basic_tissue.h"
#include "system_event_scheduler.h"
#include "legacy_heap_propagation.h"
#include "front_propagation_solver.h"

using std::vector;

struct Stimuli {
    float time;
    int beat;
    std::vector<size_t> nodes;
};

/**
 * @brief Class to model cardiac tissue. Adds propagation functions to the basic tissue.
 *
 * It contains a vector of nodes, that represent cardiac cells, and
 * the functions to perform the simulation using the fast reaction
 * diffusion model.
 */
template <typename APM,typename CVM>
class CardiacTissue : public BasicTissue<APM,CVM>
{
public:

    // using CellEvent = Event<NodeT<APM,CVM> >;
    using Node = NodeT<APM,CVM>;

    std::unique_ptr<FrontPropagationSolver<CardiacTissue<APM, CVM>>> propagation_solver;

    CardiacTissue(int size_x_, int size_y_, int size_z_, float dx_, float dy_, float dz_, PropagationSolverType solver_type = PropagationSolverType::LegacyHeap) :
        BasicTissue<APM,CVM>(size_x_, size_y_, size_z_, dx_, dy_, dz_) {
        switch (solver_type) {

            case PropagationSolverType::LegacyHeap:
                propagation_solver = std::make_unique<LegacyHeapPropagation<APM, CVM>>();
                break;
            case PropagationSolverType::FIM:
                throw std::runtime_error("CardiacTissue: FIM solver ainda nao implementado.");
        }
    }
    
    SystemEventType update(int debug = 0);

    void ScheduleActivation(const vector<size_t> & nodes, float activation_time, int beat);
    void ExternalActivation(const vector<size_t> & nodes, float activation_time, int beat_n);
    
    void Run(float end_time, std::function<void(float)> output_callback = nullptr, int debug = 0);

    // void TriggerEvent(CellEvent* ev);

    void ResetVariations() { apd_variation = 0.0; cv_variation = 0.0; }
    float GetAPDMeanVariation() const { return apd_variation / this->GetNumLiveNodes(); }
    float GetCVVariation() const { return cv_variation / this->GetNumLiveNodes(); }
    void SetLongAPDReactivation(bool val) { long_apd_reactivation = val; }

private:

    friend class LegacyHeapPropagation<APM,CVM>;

    std::vector<Stimuli> external_activations;
    void ExecuteScheduledActivation(float activation_time);

    void OnInitComplete() override;

    void SaveAdditionalState(std::ofstream& f) const override;
    void LoadAdditionalState(std::ifstream& f) override;

    void OnStateLoaded() override;

    bool long_apd_reactivation  = false;
    float apd_plateau_duration  = 0.8; // Percentage of APD considered as plateau for reactivation
    float apd_variation         = 0.0;
    float cv_variation          = 0.0;
};

template <typename APM, typename CVM>
void CardiacTissue<APM, CVM>::OnInitComplete() {
    propagation_solver->Initialize(*this);
}

template <typename APM, typename CVM>
void CardiacTissue<APM, CVM>::OnStateLoaded() {
    // LoadAdditionalState restored logical state; only reconnect pointers here.
    // propagation_solver->BindEvents(*this);
}

template <typename APM, typename CVM>
void CardiacTissue<APM, CVM>::SaveAdditionalState(std::ofstream& f) const {
    
    // Solver identifier makes the solver-specific payload explicit in the checkpoint.
    constexpr int solver_id = 1; // 1 = legacy heap propagation
    constexpr int solver_state_version = 1;
    f.write(reinterpret_cast<const char*>(&solver_id), sizeof(solver_id));
    f.write(reinterpret_cast<const char*>(&solver_state_version), sizeof(solver_state_version));

    const size_t n_stimuli = external_activations.size();
    f.write(reinterpret_cast<const char*>(&n_stimuli), sizeof(n_stimuli));
    
    for(const auto& stimulus : external_activations) {

        f.write(reinterpret_cast<const char*>(&stimulus.time), sizeof(stimulus.time));
        f.write(reinterpret_cast<const char*>(&stimulus.beat), sizeof(stimulus.beat));
        
        const size_t n_nodes = stimulus.nodes.size();
        f.write(reinterpret_cast<const char*>(&n_nodes), sizeof(n_nodes));
        
        for(size_t node_index : stimulus.nodes)
            f.write(reinterpret_cast<const char*>(&node_index), sizeof(node_index));
    }

    propagation_solver->SaveState(f, *this);
}

template <typename APM, typename CVM>
void CardiacTissue<APM, CVM>::LoadAdditionalState(std::ifstream& f) {
    
    int solver_id = 0;
    f.read(reinterpret_cast<char*>(&solver_id), sizeof(solver_id));
    
    constexpr int legacy_heap_solver_id = 1;
    if(!f || solver_id != legacy_heap_solver_id)
        throw std::runtime_error("CardiacTissue::LoadState: unsupported or invalid propagation solver identifier.");
    
    int solver_state_version = 0;
    f.read(reinterpret_cast<char*>(&solver_state_version), sizeof(solver_state_version));
    if(!f || solver_state_version != 1)
        throw std::runtime_error("CardiacTissue::LoadState: unsupported legacy solver-state version.");

    size_t n_stimuli = 0;
    f.read(reinterpret_cast<char*>(&n_stimuli), sizeof(n_stimuli));
    if(!f || n_stimuli > 100000000)
        throw std::runtime_error("CardiacTissue::LoadState: invalid external-stimulus count.");
    
    external_activations.clear();
    external_activations.reserve(n_stimuli);
    
    const size_t grid_node_count = static_cast<size_t>(this->tissue_geometry.size_x) * static_cast<size_t>(this->tissue_geometry.size_y) * static_cast<size_t>(this->tissue_geometry.size_z);
    
    for(size_t i = 0; i < n_stimuli; ++i) {

        Stimuli stimulus{};
        size_t n_nodes = 0;

        f.read(reinterpret_cast<char*>(&stimulus.time), sizeof(stimulus.time));
        f.read(reinterpret_cast<char*>(&stimulus.beat), sizeof(stimulus.beat));
        f.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));

        if(!f || n_nodes > grid_node_count)
            throw std::runtime_error("CardiacTissue::LoadState: invalid external-stimulus node count.");
        
        stimulus.nodes.resize(n_nodes);
        
        for(auto& node_index : stimulus.nodes) {
            f.read(reinterpret_cast<char*>(&node_index), sizeof(node_index));
            if(!f || node_index >= grid_node_count)
                throw std::runtime_error("CardiacTissue::LoadState: invalid external-stimulus node index.");
        }
        
        external_activations.push_back(std::move(stimulus));
    }
    
    propagation_solver->LoadState(f, *this);
}

template <typename APM, typename CVM>
void CardiacTissue<APM,CVM>::ScheduleActivation(const vector<size_t> & nodes, float activation_time, int beat) {
    this->external_activations.push_back({activation_time, beat, nodes});
    this->system_event_scheduler.Insert(activation_time, SystemEventType::EXT_ACTIVATION, 0);
}

template <typename APM, typename CVM>
void CardiacTissue<APM, CVM>::ExecuteScheduledActivation(float activation_time) {

    constexpr float time_tolerance = 1e-4f;

    for(const auto &stimulus : this->external_activations) {
        if (std::fabs(stimulus.time - activation_time) <= time_tolerance) {
            this->ExternalActivation(stimulus.nodes, stimulus.time, stimulus.beat);
            printf("Beat at time: %.2f\n", activation_time);
        }
    }

}

template <typename APM, typename CVM>
void CardiacTissue<APM,CVM>::Run(float end_time, std::function<void(float)> output_callback, int debug) {

    while (this->tissue_time < end_time) {

        SystemEventType event_type = this->update(debug);

        if(event_type == SystemEventType::NO_EVENT)
            break;

        switch (event_type) {
            case SystemEventType::EXT_ACTIVATION:
                this->ExecuteScheduledActivation(this->tissue_time);
                break;
            case SystemEventType::FILE_WRITE:
                if (output_callback) 
                    output_callback(this->tissue_time);
                break;
            default:
                break;
        }
    }
}

/**
 * Update the tissue simulation processing an event.
 * @param debug Debug level
 * @return true if there is an event of the simulation.
*/
template <typename APM,typename CVM>
SystemEventType CardiacTissue<APM,CVM>::update(int debug) {

    const bool cell_empty   = !propagation_solver->HasPendingWork();
    const bool system_empty = this->system_event_scheduler.IsEmpty();

    if(cell_empty && system_empty)
        return SystemEventType::NO_EVENT;

    bool process_system = false;

    if(cell_empty) {
        process_system = true;
    }
    else if(system_empty) {
        process_system = false;
    }
    else {
        const float cell_time = propagation_solver->NextEventTime();
        const auto &system_ev = this->system_event_scheduler.GetFirst();

        if(system_ev.event_time < cell_time) {
            process_system = true;
        }
        else if(system_ev.event_time > cell_time) {
            process_system = false;
        }
        else {
            // Same time: system event priority decides.
            process_system = system_ev.priority == 0;
        }
    }

    // ---------------------------------------------------------
    // SYSTEM EVENT
    // ---------------------------------------------------------

    if(process_system) {

        const auto &system_ev = this->system_event_scheduler.GetFirst();

        const float ev_time = system_ev.event_time;
        const SystemEventType ev_type = system_ev.type;

        LOG::Info(debug > 0, "System event at t=", ev_time, " type=", int(ev_type));

        this->tissue_time = ev_time;

        this->system_event_scheduler.ExtractFirst();

        const float inc_time = this->timer.at(int(ev_type));

        if(inc_time > 0) {

            int priority = 1;

            if(ev_type == SystemEventType::EXT_ACTIVATION)
                priority = 0;

            const float new_ev_time = this->tissue_time + inc_time;

            this->system_event_scheduler.Insert(new_ev_time, ev_type, priority);
        }

        return ev_type;
    }

    // ---------------------------------------------------------
    // CELL EVENT
    // ---------------------------------------------------------

    const float ev_time = propagation_solver->NextEventTime();
    const std::size_t node_index = propagation_solver->NextEventNodeIndex(*this);

    Node* node = &this->tissue_nodes.at(node_index);

    LOG::Info(debug > 0, "Cell event at t=", ev_time, " node=", node->id);
    LOG::Warning(ev_time < this->tissue_time, " t=", this->tissue_time, " older than ev.t=", ev_time);
    LOG::Info(debug > 1, "Before processing event. Node value: ", *node);
    LOG::Warning(node->parameters == nullptr, "Node ", node->id, " has no parameters assigned.");

    // AdvanceNext updates tissue_time, removes the event from the
    // solver queue, and processes the event.
    const std::size_t processed = propagation_solver->AdvanceNext(*this);

    if (processed == 0)
        throw std::runtime_error("CardiacTissue::update: solver reported a pending event but did not process it.");

    // Check that the solver did not leave an event in the past.
    if (propagation_solver->HasPendingWork()) {
        LOG::Error(propagation_solver->NextEventTime() < this->tissue_time, " We skipped an event!");
    }

    LOG::Info(debug > 1, "After processing event. Node value: ", *node);

    if (node->parameters != nullptr && node->parameters->sensor) {
        this->sensor_dict.AddData(node->id, node->GetData(this->tissue_time));
    }

    return SystemEventType::NODE_EVENT;
    
}

/**
 * External activation of a set of nodes.
 * @param node_ids List of node ids to activate.
 * @param activation_time Time of activation.
 *
 * @todo If the node is already active, generates a core-dump.
*/
template <typename APM,typename CVM>
void CardiacTissue<APM,CVM>::ExternalActivation(const vector<size_t> & node_ids, float activation_time, int beat_n) {
    
    for(size_t i = 0; i < node_ids.size(); i++) {

        auto node_pos = this->tissue_geometry.GetMemIndex_from_GridIndex(node_ids[i]);
        if(node_pos == NO_INDEX || this->tissue_nodes.at(node_pos).type == CELL_TYPE_VOID) {
            LOG::Warning(true, "ExternalActivation(): Node id ", node_ids[i], " is VOID or out of bounds. Activation ignored.");
            continue;
        }

        this->propagation_solver->Stimulate(*this, node_pos, activation_time, beat_n);
    }
}

// /**
//  * Process the event in the node.
//  * From Node.pde: dispara_evento
// */
// template <typename APM,typename CVM>
// void CardiacTissue<APM,CVM>::TriggerEvent(CellEvent* ev) {
//     propagation_solver->ProcessEvent(*this, ev);
// }

#include "legacy_heap_propagation_impl.h"

#endif
