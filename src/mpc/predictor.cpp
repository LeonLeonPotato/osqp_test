#include "predictor.h"
#include "osqp/osqp.h"

using namespace mpc;

std::vector<Vecf> mpc::predict(
    const Localization& localizer,
    const std::vector<Vecf>& desired_poses, 
    const Vecf& x_nom, const Vecf& u_nom, 
    const PredictParams& params)
{
    int N = std::min((int) desired_poses.size(), params.N);
    Vecf alpha; Matf beta;
    alpha_beta(desired_poses, x_nom, u_nom, params, alpha, beta);

    OSQPCscMatrix* sparse_alpha = OSQPCscMatrix_new(5*N, n, A_nnz, A_x, A_i, A_p);
}