#include "pros/rtos.hpp"
#include <iostream>
#include <vector>

#include "Eigen/Sparse"
#include "qpOASES.hpp"
#include "qpOASES/Matrices.hpp"

using namespace Eigen;
using namespace std;

MatrixXd generateDifferenceArray(int n) {
    MatrixXd diffArray(n-1, n);
    for (int i = 0; i < n-1; ++i) {
        diffArray(i, i) = 1;
        diffArray(i, i+1) = -1;
    }
    return diffArray;
}

// Main cold start test
void coldStartTest(int n) {
    const int nC = 4*n-2; // Number of constraints

    // Randomly generate the Hessian and gradient using Eigen
    MatrixXd H = MatrixXd::Random(n, n); // Hessian matrix
    VectorXd g = VectorXd::Random(n); // Gradient vector

    // Make Hessian positive definite
    H = H.transpose() * H;

    // Create constraint matrix [I, difference array]^T
    MatrixXd I = MatrixXd::Identity(n, n); // Identity matrix
    MatrixXd diffArray = generateDifferenceArray(n);

    // Stack I and difference array
    MatrixXd A(nC, n);
    A << I, diffArray.transpose();

    // Bounds (For simplicity we set them to zero bounds, as no values are given)
    VectorXd lb = VectorXd::Ones(nC) * -3; // Lower bounds
    VectorXd ub = VectorXd::Ones(nC) * 3; // Upper bounds
    VectorXd lbA = VectorXd::Ones(nC) * -12; // Lower bounds
    VectorXd ubA = VectorXd::Ones(nC) * 12; // Upper bounds

    // Initialize qpOASES problem
    qpOASES::QProblem qp(n, nC);
    
    // Time cold start
    auto start = pros::micros();
    
    // Solve the optimization problem (cold start)
    qpOASES::real_t H_qp[n * n];
    qpOASES::real_t A_qp[n * n];
    Map<Matrix<qpOASES::real_t, Dynamic, Dynamic, ColMajor>>(H_qp, H.rows(), H.cols()) = H.cast<qpOASES::real_t>();
    Map<Matrix<qpOASES::real_t, Dynamic, Dynamic, ColMajor>>(A_qp, A.rows(), A.cols()) = A.cast<qpOASES::real_t>();
    qp.init(H_qp, g.data(), A_qp, lb.data(), ub.data(), nullptr, nullptr, 10);

    auto end = pros::micros();
    auto duration = end - start;

    // Output the time taken
    cout << "Time for n = " << n << ": " << duration << " μs" << endl;
}

void test() {
    for (int i = 10; i <= 100; i += 10) {
        coldStartTest(i);
        pros::delay(100);
    }
}