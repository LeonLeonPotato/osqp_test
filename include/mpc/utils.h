#pragma once

#include "Eigen/Core"
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

namespace mpc {
using ADVec = autodiff::VectorXreal;
using Vecf =  Eigen::VectorXf;
using Matf = Eigen::MatrixXf;

struct ModelParams {
    float dt, L, gain, tc;
};

struct PredictParams {
    ModelParams model;
    int N = INT_MAX;
    Matf Q;
    Vecf R;
};

struct MPCParams {
    ModelParams model;
    int N = INT_MAX;
    float control_loop_dt;
};


ADVec diffdrive(const ADVec& x, const ADVec& u, const ModelParams& params);

void alpha_beta(
    const std::vector<Vecf>& desired_poses, 
    const Vecf& x_nom, const Vecf& u_nom, 
    const PredictParams& params,
    Vecf& alpha, Matf& beta);

// std::pair<Vecf, Eigen::MatrixXf> alpha_beta(
//     const Vecf& desired_poses,
//     const ADVec& x_nom, const ADVec& u_nom, 
//     const MPCParams& params);

std::pair<Vecf, Eigen::MatrixXf> alpha_beta(void);
}