#include "mpclib/models/diffdrive.h"
#include "mpclib/utils.h"
#include "hpipm/hpipm_s_ocp_qp.h"

using namespace mpclib;

DifferentialDriveModel::DifferentialDriveModel(const Params& params) : params_(params) {
    // Allocate general constraints matrices
    general_state_matrix = Mat::Zero(GENERAL_CONSTRAINTS_SIZE, STATE_SIZE);
    general_action_matrix = Mat::Zero(GENERAL_CONSTRAINTS_SIZE, ACTION_SIZE);

    resync_from_params();
}

// TODO: implement continuous acceleration model
ADVec DifferentialDriveModel::autodiff(const ADVec& x, const ADVec& u) const {
    ADVec xdot(5);
    auto vl = x[3] + u[0] * params_.dt;
    auto vr = x[4] + u[1] * params_.dt;
    auto ds = 0.5 * (vl + vr) * params_.dt;
    auto dtheta = (vr - vl) / params_.width * params_.dt;
    xdot[0] = x[0] + ds * ad_sinc(dtheta / 2.0) * cos(x[2] + 0.5 * dtheta);
    xdot[1] = x[1] + ds * ad_sinc(dtheta / 2.0) * sin(x[2] + 0.5 * dtheta);
    xdot[2] = x[2] + dtheta;
    xdot[3] = vl;
    xdot[4] = vr;
    return xdot;
}

Vec DifferentialDriveModel::infer(const Vec& x, const Vec& u) const {
    Vec xdot(5);
    float vl = x[3] + u[0] * params_.dt;
    float vr = x[4] + u[1] * params_.dt;
    float ds = 0.5f * (vl + vr) * params_.dt;
    float dtheta = (vr - vl) / params_.width * params_.dt;
    xdot[0] = x[0] + ds * sinc(dtheta / 2.0f) * cosf(x[2] + 0.5f * dtheta);
    xdot[1] = x[1] + ds * sinc(dtheta / 2.0f) * sinf(x[2] + 0.5f * dtheta);
    xdot[2] = x[2] + dtheta;
    xdot[3] = vl;
    xdot[4] = vr;
    return xdot;
}

void DifferentialDriveModel::resync_from_params() {
    // Set the general constraints matrices
    general_state_matrix.setZero();
    general_state_matrix(0, 3) = 1.0f;
    general_state_matrix(1, 4) = 1.0f;

    general_action_matrix.setZero();
    general_action_matrix(0, 0) = params_.acceleration_constant;
    general_action_matrix(1, 1) = params_.acceleration_constant;

    // Set the first order general constraints
    general_constraints_.clear();
    general_constraints_.reserve(GENERAL_CONSTRAINTS_SIZE);
    for (int i = 0; i < GENERAL_CONSTRAINTS_SIZE; ++i) {
        general_constraints_.emplace_back(Constraint { 
            -params_.max_speed, 
            params_.max_speed, 
            i // Actually this index parameter doesnt matter for general constraints
        });
    }
}