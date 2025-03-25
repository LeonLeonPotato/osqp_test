#pragma once

#include "localization.h"
#include "utils.h"
#include <vector>

namespace mpc {
std::vector<Vec> predict(
    const Localization& localizer,
    const std::vector<Vec>& desired_poses, 
    const Vec& x_nom, const Vec& u_nom, 
    const PredictParams& params);
}