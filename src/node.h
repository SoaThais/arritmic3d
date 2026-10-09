/**
 * ARRITMIC3D
 *
 * (C) CoMMLab-UV 2023
 * */

#ifndef NODE_H
#define NODE_H

#include <vector>
#include <limits>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

#include "definitions.h"
#include "node_parameters.h"
#include "geometry.h"

using std::vector;

constexpr int pow2(int exp) { return 1 << exp; }

template <typename ActionPotentialModel, typename ConductionVelocityModel>
class BasicTissue;
template <typename ActionPotentialModel, typename ConductionVelocityModel>
class CardiacTissue;
template <typename ActionPotentialModel, typename ConductionVelocityModel>
class LegacyHeapPropagation;

enum NodeDataId
{
    ID = pow2(0),
    TYPE = pow2(1),
    BEAT = pow2(2),
    STATE = pow2(3),
    LAT = pow2(4),
    APD = pow2(5),
    LAST_DI = pow2(6),
    CV = pow2(7),
    AP = pow2(8),
    LIFE = pow2(9),
    APD_VARIATION = pow2(10)
};




/**
 * @todo write docs
 */
template <typename ActionPotentialModel, typename ConductionVelocityModel>
class NodeT
{
public:
    using Vector3 = Eigen::Vector3f;
    using Vector2 = Eigen::Vector2f;

    friend class CardiacTissue<ActionPotentialModel, ConductionVelocityModel>;
    friend class BasicTissue<ActionPotentialModel, ConductionVelocityModel>;
    friend class LegacyHeapPropagation<ActionPotentialModel, ConductionVelocityModel>;

    /// State of the cell.
    enum class CellActivationState : char { 
        INACTIVE = 0, 
        ACTIVE 
    };

    NodeT();
    void Init(float current_time_, float initial_apd_);
    void ReApplyParam(float current_time_);
    float ComputeDirectionalConductionVelocity(const NodeT::Vector3 &direction_);
    
    unsigned int GetId() const { return id; }
    CellActivationState GetState(float current_time_) const;
    int GetBeat() const { return beat; }

    void SaveState(std::ofstream & f, const ParametersPool & parameters_pool) const;
    void LoadState(std::ifstream & f, ParametersPool & parameters_pool);

    // Data extraction ---
    // using NodeData = std::tuple<float, int, int, int, float, int, float, float, float, float, float, float>;
    using NodeData = std::tuple<float, int, int, float, int, float, float, float, float, float>;
    
    NodeData GetData(float current_time_) const {
        return NodeData(current_time_, int(type), beat, local_activation_time, apd_model.IsActive(current_time_), apd_model.getAPD(), apd_model.getLastDI(),
            conduction_vel, recovery_time, received_potential);
    }

    static const vector<std::string> GetDataNames()
    {
        static const vector<std::string> names = {"Time", "type", "beat", "local_activation_time", "activated", "APD", "LastDI", "conduction_velocity",
            "recovery_time", "received_potential"};
        return names;
    }

    //----------
    friend std::ostream & operator<<(std::ostream &os, const NodeT &node)
    {
        os << "Node id: " << node.id << " Type: " << (int)node.type << " Beat: " << node.beat;
        os << " Conduction Velocity: " << node.conduction_vel;
        os << " APD: " << node.apd_model.getAPD();
        os << " Last DI: " << node.apd_model.getLastDI();
        os << " CV: " << node.cv_model.getConductionVelocity();
        os << " LAT: " << node.local_activation_time;
        os << " Recovery time: " << node.recovery_time;
        os << " Received potential: " << node.received_potential;
        return os;
    }

private:
    constexpr static int SAVE_VERSION = 4;  ///< Version of the NodeT class for state saving/loading.
    NodeParameters* parameters;             ///< @brief Parameters of the Node
    unsigned int    id;                     ///< @brief Unique Node id. Corresponds with the grid position in the tissue.
    size_t          ext_grid_pos;           ///< @brief Position in the extended grid (index vector).

    CellType        type = CELL_TYPE_VOID;  ///< @brief Type of the Node
    bool            blocked;                ///< Node activation has been blocked
    int             beat;                   ///< @brief Last beat  of activation

    float           conduction_vel;                     ///< @brief Conduction velocity in the long. direction
    Vector3         orientation = Vector3::Zero();      ///< @brief Fiber orientation.
                                                        ///< A normalized vector indicating longitudinal direction.
                                                        ///< Default to (0,0,0) for isotropic diffusion.

    ActionPotentialModel      apd_model;

    ConductionVelocityModel   cv_model;

    float           local_activation_time; ///< @brief Time of the last activation. A.k.a. LAT.
    float           recovery_time;

    float           kapd_v;

    // Activation
    float               received_potential;

    //void Deactivate(float current_time_);
    bool Activate(float current_time_, int source_beat_, CardiacTissue<ActionPotentialModel, ConductionVelocityModel> * tissue_);
    bool ComputeActivation(float current_time_, CardiacTissue<ActionPotentialModel, ConductionVelocityModel> * tissue_);

};

#include "node_impl.h"

#endif // NODE_H
