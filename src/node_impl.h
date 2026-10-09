/**
 * ARRITMIC3D
 *
 * (C) CoMMLab-UV 2023
 * */

#include <iostream>
#include <limits>

#include "node.h"
#include "error.h"

using std::cerr;
using std::endl;


/**
 * @brief Default constructor.
 *
 */
template <typename APD, typename CVM>
NodeT<APD, CVM>::NodeT() :
    parameters(nullptr)
{ }

/**
 * @brief Obtain the activation state at the given time
 *
 * @param test_time_ Time to test the state
 * @return CellActivationState Activation state at the given time
*/
template <typename APD, typename CVM>
typename NodeT<APD, CVM>::CellActivationState NodeT<APD, CVM>::GetState(float test_time_) const {
    
    assert(test_time_ >= local_activation_time || local_activation_time == MAX_TIME);  // In the current beat

    if(this->type == CELL_TYPE_VOID)
        return CellActivationState::INACTIVE;

    if(this->apd_model.IsActive(test_time_))
        return CellActivationState::ACTIVE;

    return CellActivationState::INACTIVE;

}

/**
 * @brief Initialize the Node to the initial state.
 * From Node.pde: reset
 */
template <typename APD, typename CVM>
void NodeT<APD, CVM>::Init(float current_time_, float initial_apd_)
{
    this->beat = -1;

    // Activation data
    this->apd_model.Init(this->parameters, this->type, initial_apd_, current_time_, 0.0);
    this->cv_model.InitWithAPD(this->parameters, this->type, this->apd_model.getLastDI(), this->apd_model.getAPD() );
    this->local_activation_time = this->apd_model.getActivationTime();
    this->received_potential = 0.0;
    this->recovery_time = MAX_TIME;

    this->conduction_vel = cv_model.getConductionVelocity();
}


template <typename APD, typename CVM>
void NodeT<APD, CVM>::ReApplyParam(float current_time_)
{
    // @todo Fix: This is changing the state, not only the parameters.
    this->apd_model.Init(this->parameters, this->type,
        this->apd_model.getAPD(),
        current_time_,
        this->apd_model.getLastDI()
    );
    this->cv_model.Init(this->parameters, this->type);
}

/**
 * Calculates the action potential duration and the conduction velocity after the activation of the node.
 * From Node.pde: calcularActivacion
*/
template <typename APD, typename CVM>
bool NodeT<APD, CVM>::ComputeActivation(float current_time_,  CardiacTissue<APD, CVM> * tissue_)
{
    // Conduction velocity. Has to be activated with the previous DI. Otherwise, DI will be 0
    // Thus, we activate before updating the APD.
    // @todo Do we need electrotonic effect for CV?
    // @todo Do we need CV memory?
    bool activated = false;

    float prev_di  = this->apd_model.getDI(current_time_);
    float prev_apd = this->apd_model.getAPD();

    // Electrotonic effect
    float e_eff = this->parameters->electrotonic_effect;
    if (e_eff > 0.0) {

        float avg_apd = 0;
        unsigned int active_neighs = 0;

        for (int disp: tissue_->tissue_geometry.displacement) {
            // Danger! May go out of the array of Node
            NodeT * neigh = tissue_->NodeDisplace(this, disp);
            if(neigh == nullptr)    // We are in a void node, we skip it
                continue;
            if (neigh->GetState(current_time_) == CellActivationState::ACTIVE) {
                avg_apd += neigh->apd_model.getAPD();
                active_neighs += 1;
            }
        }

        if (active_neighs > 0) {
            avg_apd /= active_neighs;   // @todo Check: take into account *this ?
            activated = this->apd_model.Activate(current_time_, avg_apd, e_eff);
        }
        else
            activated = this->apd_model.Activate(current_time_);
    }
    else
        activated = this->apd_model.Activate(current_time_);

    if (activated)
    {
        // Conduction velocity
        this->cv_model.Activate(prev_di, prev_apd);
        this->conduction_vel = this->cv_model.getConductionVelocity( );
    }

    return activated;
}

/**
 * Activation function for the iterative case with events.
 * From Node.pde: activar
*/
template <typename APD, typename CVM>
bool NodeT<APD, CVM>::Activate(float current_time_, int source_beat_, CardiacTissue<APD, CVM> * tissue_)
{
    bool activated = false;

    if (this->GetState(current_time_) == CellActivationState::INACTIVE) {
        if(!this->ComputeActivation(current_time_, tissue_))
            activated = false;
        else {
            this->local_activation_time = current_time_;

            this->recovery_time = current_time_ + this->apd_model.getAPD();
            this->beat          = source_beat_;

            activated = true;
        }
    }

    // both if it activated or not, we reset next_activation_time
    // this->next_activation_time = MAX_TIME;
    return activated;
}

template <typename APD, typename CVM>
float NodeT<APD, CVM>::ComputeDirectionalConductionVelocity(const NodeT::Vector3 &direction_)
{
    float cond_vel;

    // If isotropic, orientation==0 or direction==0
    if(this->parameters->isotropic_diffusion ||
        this->orientation.norm() < ALMOST_ZERO ||
        direction_.norm() < ALMOST_ZERO)
    {
        // Isotropic conduction
        cond_vel = this->conduction_vel;
    }
    else if(this->type != CELL_TYPE_VOID)
    {
        // Anisotropic conduction
        float cond_vel_long = abs(this->orientation.dot(direction_))/direction_.norm();
        float cond_vel_transv = sqrt(1.0 - cond_vel_long*cond_vel_long);

        // We scale the relative position in space, according to fiber orientation.
        // Being slower in the transversal direction, space expands in that
        // direction and points are further away.
        Vector2 p;
        p[0] = cond_vel_long;
        p[1] = cond_vel_transv/this->parameters->cond_veloc_transversal_reduction;

        cond_vel = this->conduction_vel/p.norm();
    }
    else
    {
        // otherwise, VOID -> no conduction at all.
        cond_vel = 0.0;
    }

    return cond_vel;
}

template <typename APD, typename CVM>
void NodeT<APD, CVM>::SaveState(std::ofstream & f, const class ParametersPool & parameters_pool) const {
    
    f.write((char *)&SAVE_VERSION, sizeof(int));

    // Save parameters index in the pool
    size_t param_index = std::numeric_limits<size_t>::max();

    if(parameters != nullptr) {
        param_index = parameters_pool.GetIndex(parameters);
    }

    f.write((char *)&param_index, sizeof(size_t));

    // Save id
    f.write((char *)&id, sizeof(id));
    // Save extended grid position
    f.write((char *)&ext_grid_pos, sizeof(ext_grid_pos));
    // Save type
    f.write((char *)&type, sizeof(CellType));

    // Save activation data
    f.write((char *)&blocked, sizeof(blocked));
    f.write((char *)&beat, sizeof(beat));
    f.write((char *)&conduction_vel, sizeof(conduction_vel));
    f.write((char *)&orientation, sizeof(orientation));

    f.write((char *)&local_activation_time, sizeof(local_activation_time));
    f.write((char *)&kapd_v, sizeof(kapd_v));
    f.write((char *)&received_potential, sizeof(received_potential));
    f.write((char *)&recovery_time, sizeof(recovery_time));

    // Save APD model state
    apd_model.SaveState(f);

    // Save CV model state
    cv_model.SaveState(f);
}

template <typename APD, typename CVM>
void NodeT<APD, CVM>::LoadState(std::ifstream & f, ParametersPool & parameters_pool) {
    
    int version;
    f.read((char *) &version, sizeof(int));
    if(version != SAVE_VERSION)
        throw std::runtime_error("NodeT::LoadState: Wrong file version.");

    // Get the pointer to the parameters
    size_t param_index;
    f.read((char *) &param_index, sizeof(size_t));
    if(param_index != std::numeric_limits<size_t>::max())
        parameters = parameters_pool.GetParamPtr(param_index);
    else {
        parameters = nullptr;
        LOG::Warning(true, "LoadState: Node ", id, " has no parameters assigned.");
    }

    // Load id
    f.read((char *) &id, sizeof(id));
    // Load extended grid position
    f.read((char *) &ext_grid_pos, sizeof(ext_grid_pos));
    // Load type
    f.read((char *) &type, sizeof(CellType));

    // Load activation data
    f.read((char *) &blocked, sizeof(blocked));
    f.read((char *) &beat, sizeof(beat));
    f.read((char *) &conduction_vel, sizeof(conduction_vel));
    f.read((char *) &orientation, sizeof(orientation));

    f.read((char *) &local_activation_time, sizeof(local_activation_time));
    f.read((char *) &kapd_v, sizeof(kapd_v));
    f.read((char *) &received_potential, sizeof(received_potential));
    f.read((char *) &recovery_time, sizeof(recovery_time));

    // Load APD model state
    apd_model.LoadState(f, type, parameters);

    // Load CV model state
    cv_model.LoadState(f, type, parameters);
}


