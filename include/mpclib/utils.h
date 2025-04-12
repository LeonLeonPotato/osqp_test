#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

namespace mpclib {
using type = float;
using ADVec = autodiff::VectorXreal;
using Vec =  Eigen::VectorX<type>;
using Mat = Eigen::MatrixX<type>;
using SPMat = Eigen::SparseMatrix<type>;

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
static inline T sinc(T x) {
    if (std::abs(x) < 1e-3) return 1 - (x*x/6.0f) + (x*x*x*x/120.0f);
    return std::sin(x) / x;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
static inline float average(const std::vector<T>& vec) {
    if (vec.empty()) return 0;

    // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    double sum = 0; double z = 0;
    for (const auto& val : vec) {
        double y = val - z;
        double t = sum + y;
        z = (t - sum) - y;
        sum = t;
    }
    return static_cast<float>(sum / vec.size());
}

static inline float modfix(float x, float y) {
    float result = fmodf(x, y);
    return result + (result < 0) * y;
}
}