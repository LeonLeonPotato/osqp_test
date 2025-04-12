#include "mpclib/models/diffdrive.h"
#include "mpclib/utils.h"
#include "hpipm/hpipm_s_ocp_qp.h"

using namespace mpclib;

DifferentialDriveModel::DifferentialDriveModel(const Params& params) : params(params) {
    // We actually do not want to implement velocity box constraints on the state variables
    // This will be handled via the general constraints (which allows me to implement the first order system behavior)
    state_lower_bound_mask[3] = 0;
    state_lower_bound_mask[4] = 0;
    state_upper_bound_mask[3] = 0;
    state_upper_bound_mask[4] = 0;

    // At the same time we don't want to constrain acceleration as this can result in inconsistent constraints
    action_lower_bound_mask[0] = 0;
    action_lower_bound_mask[1] = 0;
    action_upper_bound_mask[0] = 0;
    action_upper_bound_mask[1] = 0;

    // Allocate general constraints matrices
    general_constraints_state_matrix = Mat::Zero(GENERAL_CONSTRAINTS_SIZE, STATE_SIZE);
    general_constraints_action_matrix = Mat::Zero(GENERAL_CONSTRAINTS_SIZE, ACTION_SIZE);
}

// TODO: implement continuous acceleration model
ADVec DifferentialDriveModel::autodiff(const ADVec& x, const ADVec& u) const {
    ADVec xdot(5);
    auto vl = x[3] + u[0] * params.dt;
    auto vr = x[4] + u[1] * params.dt;
    auto ds = 0.5 * (vl + vr) * params.dt;
    auto dtheta = (vr - vl) / params.width * params.dt;
    xdot[0] = x[0] + ds * ad_sinc(dtheta / 2.0) * cos(x[2] + 0.5 * dtheta);
    xdot[1] = x[1] + ds * ad_sinc(dtheta / 2.0) * sin(x[2] + 0.5 * dtheta);
    xdot[2] = x[2] + dtheta;
    xdot[3] = vl;
    xdot[4] = vr;
    return xdot;
}

Vec DifferentialDriveModel::infer(const Vec& x, const Vec& u) const {
    Vec xdot(5);
    float vl = x[3] + u[0] * params.dt;
    float vr = x[4] + u[1] * params.dt;
    float ds = 0.5f * (vl + vr) * params.dt;
    float dtheta = (vr - vl) / params.width * params.dt;
    xdot[0] = x[0] + ds * sinc(dtheta / 2.0f) * cosf(x[2] + 0.5f * dtheta);
    xdot[1] = x[1] + ds * sinc(dtheta / 2.0f) * sinf(x[2] + 0.5f * dtheta);
    xdot[2] = x[2] + dtheta;
    xdot[3] = vl;
    xdot[4] = vr;
    return xdot;
}

float* DifferentialDriveModel::get_general_lower_bound() {
    general_lower_bound[0] = -params.max_speed;
    general_lower_bound[1] = -params.max_speed;
    return general_lower_bound;
}

float* DifferentialDriveModel::get_general_upper_bound() {
    general_upper_bound[0] = params.max_speed;
    general_upper_bound[1] = params.max_speed;
    return general_upper_bound;
}

float* DifferentialDriveModel::get_general_constraints_state_matrix() {
    general_constraints_state_matrix(0, 3) = 1.0f;
    general_constraints_state_matrix(1, 4) = 1.0f;
    return general_constraints_state_matrix.data();
}

float* DifferentialDriveModel::get_general_constraints_action_matrix() {
    general_constraints_action_matrix(0, 0) = params.acceleration_constant;
    general_constraints_action_matrix(1, 1) = params.acceleration_constant;
    return general_constraints_action_matrix.data();
}