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
    long long gstart = pros::micros();
    int N = 25;

    mpc::PredictParams params {{0.03, 30.54, 18.674, 0.1626}, N};
    params.Q = Eigen::Vector<float, 5> {1, 1, 1, 0, 0}.asDiagonal();
    params.R = Eigen::MatrixXf::Identity(2, 2) * 0.0f;
    Eigen::VectorXf x_nom(5);
    x_nom << 1, -1, M_PI_2 + 0.01, 5, 4.9;
    Eigen::VectorXf u_nom(2);
    u_nom << 1, -1;

    mpc::Vec target (5);
    target << 0, 100, 0, 0, 0;
    std::vector<Eigen::VectorXf> desired_poses(N, target);
    mpc::SimulatedLocalizer localizer(0, 0, 0);

    // mpc::predict(localizer, desired_poses, x_nom, u_nom, params);
    mpc::SPMat H;
    mpc::Vec q;
    long long start1 = pros::micros();
    mpc::compute_penalties(desired_poses, x_nom, u_nom, params, H, q);
    long long end1 = pros::micros();
    std::cout << "Penalty time creation: " << end1 - start1 << " us" << std::endl;

    // std::cout << "H: \n" << H.toDense() << std::endl;
    // std::cout << "q: \n" << q << std::endl;
    Eigen::SparseMatrix<float> eye (4*N - 2, 2*N);
    for (int i = 0; i < 2*N; i++) eye.insert(i, i) = 1.0f;
    for (int i = 0; i < 2*N - 2; i++) {
        eye.insert(i + 2*N, i) = 1.0f;
        eye.insert(i + 2*N, i+2) = -1.0f;
    }
    mpc::Vec lower_bound(4*N - 2);
    lower_bound.head(2*N).setConstant(-12);
    lower_bound.tail(2*N).setConstant(-1);
    mpc::Vec upper_bound(4*N - 2);
    upper_bound.head(2*N).setConstant(12);
    upper_bound.tail(2*N).setConstant(1);

    // printf("Setting up solver\n");
    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(true);
    solver.settings()->setMaxIteration(50);
    solver.data()->setNumberOfConstraints(4*N - 2);
    solver.data()->setNumberOfVariables(2*N);
    solver.data()->setHessianMatrix(H);
    solver.data()->setGradient(q);
    solver.data()->setLinearConstraintsMatrix(eye);
    solver.data()->setLowerBound(lower_bound);
    solver.data()->setUpperBound(upper_bound);

    solver.initSolver();
    long long start2 = pros::micros();
    auto flag = solver.solveProblem();
    long long end2 = pros::micros();
    // printf("Flag: %d\n", (int) flag);
    if (flag != OsqpEigen::ErrorExitFlag::NoError) return;

    mpc::Vec solution = solver.getSolution();

    std::cout << "Solution:\n" << solution << std::endl;
    std::cout << "Solve time: " << end2 - start2 << " us" << std::endl;

    long long gend = pros::micros();
    std::cout << "Total time: " << gend - gstart << " us" << std::endl;
}