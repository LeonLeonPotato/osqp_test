#pragma once

#include "Eigen/Core"
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

namespace mpc {
using ADVec = autodiff::VectorXreal;

struct ModelParams {
    float dt, L, gain, tc;
};

struct MPCParams {
    ModelParams model;
    float control_loop_dt;
    int N;
};


ADVec diffdrive(const ADVec& x, const ADVec& u, const ModelParams& params);

std::pair<Eigen::VectorXf, Eigen::MatrixXf> alpha_beta(
    const std::vector<Eigen::VectorXf>& desired_poses, 
    const Eigen::VectorXf& x_nom, const Eigen::VectorXf& u_nom, 
    const MPCParams& params);

// std::pair<Eigen::VectorXf, Eigen::MatrixXf> alpha_beta(
//     const Eigen::VectorXf& desired_poses,
//     const ADVec& x_nom, const ADVec& u_nom, 
//     const MPCParams& params);

std::pair<Eigen::VectorXf, Eigen::MatrixXf> alpha_beta(void);
}