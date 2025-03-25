#include "autodiff/forward/utils/gradient.hpp"
#include "mpc/localization.h"
#include "mpc/predictor.h"
#include "mpc/utils.h"
#include "osqp/osqp_api_functions.h"
#include "pros/rtos.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "osqp/osqp.h"

void test() {
    int N = 3;

    mpc::PredictParams params {{0.03, 30.54, 18.674, 0.1626}, N};
    params.Q = Eigen::MatrixXf::Ones(5, 5);
    params.R = Eigen::MatrixXf::Ones(2, 2);
    Eigen::VectorXf x_nom(5);
    x_nom << 1, -1, 2, 1, 1;
    Eigen::VectorXf u_nom(2);
    u_nom << 1, -1;

    std::vector<Eigen::VectorXf> desired_poses(N, x_nom.cast<float>());
    mpc::SimulatedLocalizer localizer(0, 0, 0);

    // mpc::predict(localizer, desired_poses, x_nom, u_nom, params);
    mpc::Matf H;
    mpc::Vecf q;
    long long start = pros::micros();
    mpc::compute_penalties(desired_poses, x_nom, u_nom, params, H, q);
    long long end = pros::micros();

    std::cout << "H: \n" << H << std::endl;
    std::cout << "q: \n" << q << std::endl;

    std::cout << "OSQP test time taken: " << end - start << " us" << std::endl;
}