#include "predictor.h"
#include "osqp/osqp.h"

using namespace mpc;

std::vector<Vecf> mpc::predict(
    const Localization& localizer,
    const std::vector<Vecf>& desired_poses, 
    const Vecf& x_nom, const Vecf& u_nom, 
    const PredictParams& params)
{
    Vecf alpha; Matf beta;
    alpha_beta(desired_poses, x_nom, u_nom, params, alpha, beta);

    
}