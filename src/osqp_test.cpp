#include "autodiff/forward/utils/gradient.hpp"
#include "mpc/utils.h"
#include "pros/rtos.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "osqp/osqp.h"

void test() {
    int N = 100;

    long long start = pros::micros();

    mpc::MPCParams params {{0.03, 30.54, 18.674, 0.1626}, 0.01, N};
    Eigen::VectorXf x_nom(5);
    x_nom << 1, -1, 2, 1, 1;
    Eigen::VectorXf u_nom(2);
    u_nom << 1, -1;

    std::vector<Eigen::VectorXf> desired_poses(N, x_nom.cast<float>());

    auto ret = mpc::alpha_beta(desired_poses, x_nom, u_nom, params);
    long long end = pros::micros();

    std::cout << "Time taken: " << end - start << " us" << std::endl;
}