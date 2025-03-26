#include "predictor.h"
#include "utils.h"
#include <vector>

using namespace mpc;

std::vector<Vec> mpc::predict(
    const Localization& localizer,
    const std::vector<Vec>& desired_poses, 
    const Vec& x_nom, const Vec& u_nom, 
    const PredictParams& params)
{
    // long long start = pros::micros();
    // int N = std::min((int) desired_poses.size(), params.N);
    // Vecf alpha; Matf beta; 
    // Matf* betaTQ = new Matf();
    // alpha_beta(desired_poses, x_nom, u_nom, params, alpha, beta, betaTQ);

    // Matf H = betaTQ->operator*(beta);
    // Vecf q = betaTQ->operator*(alpha);
    // long long end = pros::micros();
    // std::cout << "Predictor took " << end - start << " us" << std::endl;
    
    std::vector<Vec> predictions;
    return predictions;
}