#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

namespace mpc {
using type = float;
using ADVec = autodiff::VectorXreal;
using Vec =  Eigen::VectorX<float>;
using Mat = Eigen::MatrixX<float>;
using SPMat = Eigen::SparseMatrix<float>;

struct ModelParams {
    float dt, L, gain, tc;
};

struct PredictParams {
    ModelParams model;
    int N = INT_MAX;
    Mat Q;
    Mat R;
};

struct MPCParams {
    ModelParams model;
    int N = INT_MAX;
    float control_loop_dt;
};


ADVec diffdrive(const ADVec& x, const ADVec& u, const ModelParams& params);

void alpha_beta(
    const std::vector<Vec>& desired_poses, 
    const Vec& x_nom, const Vec& u_nom, 
    const PredictParams& params,
    Vec& alpha, Mat& beta, Mat* betaTQ = nullptr);

void compute_penalties(
    const std::vector<Vec>& desired_poses, 
    const Vec& x_nom, const Vec& u_nom, 
    const PredictParams& params,
    SPMat& H, Vec& q);

// std::pair<Vecf, Eigen::MatrixXf> alpha_beta(
//     const Vecf& desired_poses,
//     const ADVec& x_nom, const ADVec& u_nom, 
//     const MPCParams& params);

std::pair<Vec, Eigen::MatrixXf> alpha_beta(void);
}