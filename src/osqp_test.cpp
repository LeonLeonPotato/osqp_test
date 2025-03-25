#include "OsqpEigen/Constants.hpp"
#include "OsqpEigen/OsqpEigen.h"
#include "OsqpEigen/Solver.hpp"
#include "mpc/localization.h"
#include "mpc/predictor.h"
#include "mpc/utils.h"
#include "pros/rtos.hpp"
#include <iostream>
#include <vector>

#include "Eigen/Sparse"

void test() {
    int N = 3;

    mpc::PredictParams params {{0.03, 30.54, 18.674, 0.1626}, N};
    params.Q = Eigen::MatrixXf::Identity(5, 5);
    params.R = Eigen::MatrixXf::Identity(2, 2);
    Eigen::VectorXf x_nom(5);
    x_nom << 1, -1, 2, 0, 0;
    Eigen::VectorXf u_nom(2);
    u_nom << 0, 0;

    std::vector<Eigen::VectorXf> desired_poses(N, x_nom.cast<float>());
    mpc::SimulatedLocalizer localizer(0, 0, 0);

    // mpc::predict(localizer, desired_poses, x_nom, u_nom, params);
    mpc::SPMat H;
    mpc::Vec q;
    long long start = pros::micros();
    mpc::compute_penalties(desired_poses, x_nom, u_nom, params, H, q);
    long long end = pros::micros();

    std::cout << "H: \n" << H << std::endl;
    std::cout << "q: \n" << q << std::endl;
    Eigen::SparseMatrix<float> eye (2*N, 2*N);
    for (int i = 0; i < 2*N; i++) eye.insert(i, i) = 1.0f;
    mpc::Vec lower_bound(2*N);
    lower_bound.setConstant(-12);
    mpc::Vec upper_bound(2*N);
    upper_bound.setConstant(12);

    printf("Setting up solver\n");
    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(true);
    solver.data()->setNumberOfConstraints(2*N);
    solver.data()->setNumberOfVariables(2*N);
    solver.data()->setHessianMatrix(H);
    solver.data()->setGradient(q);
    solver.data()->setLinearConstraintsMatrix(eye);
    solver.data()->setLowerBound(lower_bound);
    solver.data()->setUpperBound(upper_bound);

    solver.initSolver();
    auto flag = solver.solveProblem();
    printf("Flag: %d\n", (int) flag);
    if (flag != OsqpEigen::ErrorExitFlag::NoError) return;

    mpc::Vec solution = solver.getSolution();
    std::cout << "Solution:\n" << solution << std::endl;

    std::cout << "OSQP test time taken: " << end - start << " us" << std::endl;
}