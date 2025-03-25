#include "mpc/utils.h"
#include "Eigen/src/Core/NumTraits.h"
#include "Eigen/src/SparseCore/SparseUtil.h"
#include "api.h"
#include <utility>

using namespace mpc;

ADVec mpc::diffdrive(const ADVec &x, const ADVec &u, const ModelParams& params) {
    ADVec xdot(5);
    auto vl = (params.gain * u[0] - x[3]) / params.tc * params.dt + x[3];
    auto vr = (params.gain * u[1] - x[4]) / params.tc * params.dt + x[4];
    auto ds = 0.5 * (vl + vr) * params.dt;
    auto dtheta = (vr - vl) / params.L * params.dt;
    xdot[0] = x[0] + ds * cos(x[2] + 0.5 * dtheta);
    xdot[1] = x[1] + ds * sin(x[2] + 0.5 * dtheta);
    xdot[2] = x[2] + dtheta;
    xdot[3] = vl;
    xdot[4] = vr;
    return xdot;
}

void mpc::alpha_beta(
    const std::vector<Vec>& desired_poses, 
    const Vec& x_nom, const Vec& u_nom, 
    const PredictParams& params,
    Vec& alpha, Mat& beta, Mat* betaTQ)
{
    int N = std::min(params.N, (int) desired_poses.size());

    ADVec fout;
    ADVec ad_x_nom = x_nom.cast<autodiff::real>();
    ADVec ad_u_nom = u_nom.cast<autodiff::real>();
    Mat jx = autodiff::jacobian(diffdrive, autodiff::wrt(ad_x_nom), autodiff::at(ad_x_nom, ad_u_nom, params.model), fout).cast<float>();
    Mat ju = autodiff::jacobian(diffdrive, autodiff::wrt(ad_u_nom), autodiff::at(ad_x_nom, ad_u_nom, params.model), fout).cast<float>();
    Vec c = fout.cast<float>() - jx * x_nom - ju * u_nom;

    Mat A(N*5, 5);
    A.block(0, 0, 5, 5).setIdentity(); // Set the first block to the identity matrix, which is jx^0
    Vec B2C(N*5);
    B2C.segment(0, 5) = c; // This is equal to the first block of A * c, which is equal to c

    for (int i = 1; i < N; i++) {
        A.block(5*i, 0, 5, 5) = A.block(5*(i-1), 0, 5, 5) * jx;
        B2C.segment(5*i, 5) = B2C.segment(5*(i-1), 5) + A.block(5*i, 0, 5, 5) * c;
    }

    bool use_betaTQ = betaTQ != nullptr;
    if (use_betaTQ) betaTQ->resize(N*2, N*5);
    beta.resize(N*5, N*2);
    for (int i = 0; i < N; i++) {
        Mat fill = A.block(5*i, 0, 5, 5) * ju;

        for (int j = 0; j < N - i; j++) {
            int row = (i+j)*5, col = j*2;
            beta.block(row, col, 5, 2) = fill;
            if (use_betaTQ) {
                Mat fillTQ = fill.transpose() * params.Q;
                betaTQ->block(col, row, 2, 5) = fillTQ;
            }

            if (i != 0) {
                row = j*5; col = (i+j)*2;
                beta.block(row, col, 5, 2).setZero();
                if (use_betaTQ) {
                    betaTQ->block(col, row, 2, 5).setZero();
                }
            }
        }
    }

    Vec stacked_desired_poses(5*N);
    for (int i = 0; i < N; i++) {
        stacked_desired_poses.segment(5*i, 5) = desired_poses[i];
    }
    alpha = A * jx * x_nom + B2C - stacked_desired_poses;
}

void mpc::compute_penalties(
    const std::vector<Vec>& desired_poses, 
    const Vec& x_nom, const Vec& u_nom, 
    const PredictParams& params,
    SPMat& H, Vec& q)
{
    int N = std::min(params.N, (int) desired_poses.size());
    if (N <= 0) {
        printf("[MPC] WARNING: N is 0 or negative, skipping penalty computation! This is a catastrophic error, \
                please check your code to make sure you input the right values!\n");
        H.diagonal().setOnes();
        q.setOnes();
        return;
    }

    ADVec fout;
    ADVec ad_x_nom = x_nom.cast<autodiff::real>();
    ADVec ad_u_nom = u_nom.cast<autodiff::real>();
    Mat jx = autodiff::jacobian(diffdrive, autodiff::wrt(ad_x_nom), autodiff::at(ad_x_nom, ad_u_nom, params.model), fout).cast<float>();
    Mat ju = autodiff::jacobian(diffdrive, autodiff::wrt(ad_u_nom), autodiff::at(ad_x_nom, ad_u_nom, params.model), fout).cast<float>();
    Vec c = fout.cast<float>() - jx * x_nom - ju * u_nom;

    // We will be using Jx^0 (identity matrix) all the way to Jx^N
    // So, we will compute them all here and use blocks to select portions
    Mat A((N+1)*5, 5);
    A.block(0, 0, 5, 5).setIdentity(); // Set the first block to the identity matrix, which is Jx^0
    Vec B2C(N*5);
    B2C.head(5) = c; // This is equal to the first block of A * c, which is equal to c

    for (int i = 1; i < N; i++) {
        // Compute Jx^i by multiplying Jx^(i-1) by Jx
        A.block(5*i, 0, 5, 5) = A.block(5*(i-1), 0, 5, 5) * jx;
        // Compute B2C by adding Jx^i * c to the previous element of B2C
        // This computes the sum of all Jx^j * c for j = 0 to i
        B2C.segment(5*i, 5) = B2C.segment(5*(i-1), 5) + A.block(5*i, 0, 5, 5) * c;
    }
    // The last block of A is Jx^N, which is the same as Jx^(N-1) * Jx
    A.block(5*N, 0, 5, 5) = A.block(5*(N-1), 0, 5, 5) * jx;

    Mat betaTQ(2, N*5); // This is a cache of Ju^T * Jx^i^T * Q for i from 0 to N-1
    Mat beta(N*5, 2); // This is a cache of Jx^i * Ju for i from 0 to N-1
    Mat dp(N*2, 2); // dynamic programming solution

    for (int i = 0; i < N; i++) {
        Mat fill = A.block(5*i, 0, 5, 5) * ju;
        beta.block(5*i, 0, 5, 2) = fill;
        betaTQ.block(0, 5*i, 2, 5) = fill.transpose() * params.Q;
    }

    for (int i = N-1; i >= 0; i--) {
        dp.block(2*i, 0, 2, 2) = betaTQ.block(0, 5*i, 2, 5) * beta.block(5*i, 0, 5, 2);
        if (i != N-1) {
            dp.block(2*i, 0, 2, 2) += dp.block(2*i+2, 0, 2, 2);
        }
    }
    
    std::vector<Eigen::Triplet<float>> triplets;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            int dstrow = i*2, dstcol = j*2;
            Mat blk = dp.block(2*j, 0, 2, 2);
            if (i == j) {
                blk += params.R;
            }


            for (int k = 0; k < blk.rows(); k++) {
                for (int l = 0; l < blk.cols(); l++) {
                    if (dstrow + k > dstcol + l) continue;
                    if (fabsf(blk(k, l)) > Eigen::NumTraits<float>::dummy_precision()) {
                        float val = blk(k, l);
                        triplets.emplace_back(dstrow + k, dstcol + l, val);
                    }
                }
            }
        }
    }
    H.resize(N*2, N*2);
    H.setFromTriplets(triplets.begin(), triplets.end());

    Mat transposed_jx = jx.transpose().eval();
    Mat transposed_ju = ju.transpose().eval();

    #define alpha(i) (A.block(5*(i), 0, 5, 5) * x_nom + B2C.segment(5*((i) - 1), 5) - desired_poses[(i) - 1])

    Vec q_intermediate(5*N); // DP way of computing q, except without Ju^T at the front
    q_intermediate.tail(5) = params.Q * alpha(N); // These past 3 lines calculate alpha_N
    q.resize(2*N);
    q.tail(2) = transposed_ju * q_intermediate.tail(5);

    for (int i = N-2; i >= 0; i--) {
        q_intermediate.segment(5*i, 5) = transposed_jx 
            * q_intermediate.segment(5*(i+1), 5) // because of the pattern of the resultant vector, this shortcut calculation works
            + params.Q * alpha(i+1); // calculate alpha_i
        q.segment(2*i, 2) = transposed_ju * q_intermediate.segment(5*i, 5);
    }
}