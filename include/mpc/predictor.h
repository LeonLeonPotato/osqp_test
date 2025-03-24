#pragma once

#include "localization.h"
#include "utils.h"
#include <vector>

namespace mpc {
std::vector<Vecf> predict(
    const Localization& localizer,
    const std::vector<Vecf>& desired_poses, 
    const Vecf& x_nom, const Vecf& u_nom, 
    const PredictParams& params);
}