#include "mpclib/models/diffdrive.h"
#include "mpclib/utils.h"

using namespace mpclib;

static auto inline ad_sinc(autodiff::real x) {
    if (abs(x) < 1e-3) return autodiff::real(1) - (x*x/6.0f) + (x*x*x*x/120.0f);
    return sin(x) / x;
}

ADVec DifferentialDriveModel::autodiff(const ADVec& x, const ADVec& u) const {
    ADVec xdot(5);
    auto vl = x[3] + u[0] * params.dt;
    auto vr = x[4] + u[1] * params.dt;
    auto ds = 0.5 * (vl + vr) * params.dt;
    auto dtheta = (vr - vl) / params.width * params.dt;
    xdot[0] = x[0] + ds * ad_sinc(dtheta / 2) * cos(x[2] + 0.5 * dtheta);
    xdot[1] = x[1] + ds * ad_sinc(dtheta / 2) * sin(x[2] + 0.5 * dtheta);
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
    xdot[0] = x[0] + ds * sinc(dtheta / 2) * cosf(x[2] + 0.5f * dtheta);
    xdot[1] = x[1] + ds * sinc(dtheta / 2) * sinf(x[2] + 0.5f * dtheta);
    xdot[2] = x[2] + dtheta;
    xdot[3] = vl;
    xdot[4] = vr;
    return xdot;
}

float* DifferentialDriveModel::get_state_lower_bound() {
    state_lower_bound[3] = -params.max_speed;
    state_lower_bound[4] = -params.max_speed;
    return state_lower_bound;
}

float* DifferentialDriveModel::get_state_upper_bound() {
    state_upper_bound[3] = params.max_speed;
    state_upper_bound[4] = params.max_speed;
    return state_upper_bound;
}

float* DifferentialDriveModel::get_action_lower_bound() {
    action_lower_bound[0] = -params.max_acceleration;
    action_lower_bound[1] = -params.max_acceleration;
    return action_lower_bound;
}

float* DifferentialDriveModel::get_action_upper_bound() {
    action_upper_bound[0] = params.max_acceleration;
    action_upper_bound[1] = params.max_acceleration;
    return action_upper_bound;
}

float* DifferentialDriveModel::get_general_lower_bound() {
    general_lower_bound[0] = -params.max_speed;
    general_lower_bound[1] = -params.max_speed;
    return general_lower_bound;
}