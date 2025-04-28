#pragma once

#include "polynomial.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <queue>
#include <stack>

#include "Eigen/Sparse" // IWYU pragma: keep

#define __DEFINE_FORWARDER(name, type) \
    template <int N> \
    type NthDegreeSpline<N>::name(float t) const { \
        const int i = i_helper(t); \
        return segments[i].name(t); \
    }

namespace pathing {
struct Condition {
    int derivative; 
    float theta, magnitude;

    Eigen::Vector2f cartesian() const {
        return Eigen::Vector2f {sinf(theta) * magnitude, cosf(theta) * magnitude};
    }

    float x() const { return sinf(theta) * magnitude; }
    float y() const { return cosf(theta) * magnitude; }

    static Condition from_cartesian(const int deriv, const Eigen::Vector2f& p) {
        return {deriv, atan2f(p.x(), p.y()), p.norm()};
    }

    static Condition from_cartesian(const int deriv, const float x, const float y) {
        return {deriv, atan2f(x, y), sqrtf(x*x + y*y)};
    }
};

struct ProfileParams {
    float max_speed, max_accel, min_speed;
    float track_width;

    float dt;
};

struct ProfilePoint {
    struct ConstructionPoint {
        float speed, distance;
    }; ///< Partial profile point for construction of profile, less memory usage
    
    Eigen::Vector2f pos; ///< position given path parameter
    Eigen::Vector2f deriv1st; ///< first derivative of position w.r.t path parameter
    Eigen::Vector2f deriv2nd; ///< second derivative of position w.r.t path parameter
    
    float distance; ///< Arc length from the start of the path to this point
    float speed; ///< Theoretical speed (w.r.t. real world time) at this point
    float accel; ///< Theoretical acceleration (w.r.t. real world time) at this point

    float path_param; ///< The path parameter at this point
    float real_time; ///< Theoretical real world time it took to get to this point

    Eigen::Vector2f get_track_speeds(const ProfileParams& params) const {
        float norm = 1.0f / (deriv1st.squaredNorm() * deriv1st.norm() + 1e-6f);
        float scalar_denom = deriv1st.cross(deriv2nd) * norm * params.track_width * 0.5f;
        float left = speed * (1.0f - scalar_denom);
        float right = speed * (1.0f + scalar_denom);
        return {left, right};
    }

    auto operator<(const float& other) const { return real_time < other; }
};


template <int N>
class NthDegreeSpline {
    private:
        std::vector<Eigen::Vector2f> points;
        std::vector<std::pair<float, float>> lengths;
        std::vector<ProfilePoint> profile;

        static std::vector<float> zerodiffs;
        static Eigen::MatrixXf pascal;
        static void ensure(int n);

        std::vector<Polynomial2D<N>> segments;

        int i_helper(float& t) const;
        float get_pascal_coefficient(int i, int j) const;
        void solve_spline(int axis, 
            const std::vector<Condition>& ics, 
            const std::vector<Condition>& bcs);
        
    public:
        static const std::vector<Condition> natural_conditions;

        NthDegreeSpline(void) { ensure(N); }
        NthDegreeSpline(int n) { segments.resize(n); ensure(N);}
        NthDegreeSpline(const std::vector<Eigen::Vector2f>& verts) { points = verts; ensure(N); }

        int maxt() const { return points.size() - 1; }
        float length(float t) const;
        float path_parametrize(float s) const;
        void profile_path(const ProfileParams& params);
        const std::vector<ProfilePoint>& get_profile() const { return profile; }

        void solve_coeffs(
            const std::vector<Condition>& ics, 
            const std::vector<Condition>& bcs);
        
        void compute(float t, Eigen::Vector2f& res, int deriv = 0) const;
        Eigen::Vector2f compute(float t, int deriv = 0) const;

        Eigen::Vector2f normal(float t) const;
        float angle(float t) const;
        float angular_velocity(float t) const;
        float curvature(float t) const;

        std::string debug_out(void) const;
        std::string debug_out_precise(int precision = 4) const;

        template <typename... Args>
        auto operator()(Args... args) -> decltype(compute(args...)) const { return compute(args...); }
};

template <int N>
inline Eigen::MatrixXf NthDegreeSpline<N>::pascal = Eigen::MatrixXf::Ones(1, 1);

template <int N>
inline std::vector<float> NthDegreeSpline<N>::zerodiffs = {1};

template <int N>
inline const std::vector<Condition> NthDegreeSpline<N>::natural_conditions = []() {
    std::vector<Condition> cs;
    for (int i = 0; i < N/2; i++) {
        cs.push_back({i+2, 0, 0});
    }
    return cs;
}();

template<int N>
void NthDegreeSpline<N>::ensure(int n) {
    if (n % 2 == 0) {
        std::cerr << "N (Currently " << n << ") must be odd, do not use even-degreed splines!\n";
        return;
    }

    if (pascal.rows() >= n) return;

    pascal.resize(n, n);
    pascal.row(0).setConstant(1.0f);
    pascal.col(0).setConstant(1.0f);
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n - i; j++) {
            pascal(i, j) = pascal(i-1, j) + pascal(i, j-1);
        }
    }

    zerodiffs.resize(n+1);
    for (int i = 0; i < n+1; i++) {
        zerodiffs[i] = 1;
        for (int j = i; j > 0; j--) zerodiffs[i] *= j;
    }
}

template <int N>
inline int NthDegreeSpline<N>::i_helper(float& t) const {
    t = std::clamp(t, 0.0f, (float) segments.size());
    const int i = (int) t - (int) (t == segments.size());
    t -= i;
    return i;
}

template<int N>
float NthDegreeSpline<N>::get_pascal_coefficient(int i, int j) const {
    if (j >= N - i) return j % 2 == 0 ? -1 : 1;
    return NthDegreeSpline::pascal(i, j);
}

template<int N>
void NthDegreeSpline<N>::solve_spline(int axis, 
            const std::vector<Condition>& ics_og, 
            const std::vector<Condition>& bcs_og)
{
    // Sanitize and setup
    auto comp = [](const Condition& a, const Condition& b) {
        return a.derivative < b.derivative;
    };
    std::vector<Condition> ics = ics_og; std::sort(ics.begin(), ics.end(), comp);
    std::vector<Condition> bcs = bcs_og; std::sort(bcs.begin(), bcs.end(), comp);

    // Useful constants
    const int ics_length = ics.size();
    const int bcs_length = bcs.size();
    // Originally, we have N * segments.size() unknowns, but we "collapse" the ICs into the initial rows
    int n = N * segments.size() - ics_length;
    int half = N / 2;
    int max_pivot_find = std::min(half, 3); 
    int eff_length = half + max_pivot_find + 1;

    if (ics_length != half) {
        std::cerr << "Invalid number of ICs: Expected " << half << ", but got " << ics_length << std::endl;
        return;
    }

    if (bcs_length != half) {
        std::cerr << "Invalid number of BCs: Expected " << half << ", but got " << bcs_length << std::endl;
        return;
    }

    if (ics_length > 0 && ics[0].derivative < 1) {
        std::cerr << "ICs must start at least at the first derivative! Current smallest derivative: %d" << ics[0].derivative << "\n";
        return;
    }

    if (bcs_length > 0 && bcs[0].derivative < 1) {
        std::cerr << "BCs must start at least at the first derivative! Current smallest derivative: %d" << bcs[0].derivative << "\n";
        return;
    }

    if (segments.size() == 1) { // For some reason this is a special case
        Eigen::MatrixXf A(N, N); A.setZero();
        Eigen::VectorXf B(N); B.setZero();

        for (int i = 0; i < ics_length; i++) {
            A(i, ics[i].derivative - 1) = zerodiffs[ics[i].derivative];
            B[i] = ics[i].cartesian()[axis];
        }

        A.row(half).setConstant(1.0f);
        B[half] = points[1][axis] - points[0][axis];
        for (int i = 0; i < bcs_length; i++) {
            B[N-1-i] = bcs[i].cartesian()[axis];
            // cast to float as well
            A.row(N-1-i) = Polynomial<N>::get_differential(N).col(bcs[i].derivative).tail(N).template cast<float>();
        }

        Eigen::VectorXf X = A.colPivHouseholderQr().solve(B);

        auto& poly = axis == 0 ? segments[0].x_poly : segments[0].y_poly;
        for (int i = 0; i < N; i++) {
            poly.coeffs[i+1] = X[i];
        }
        poly.coeffs[0] = points[0][axis];
        return;
    }

    // IC Collapse preparation
    bool ic_flags[N]; std::fill(ic_flags, ic_flags + N, false);
    float ic_vals[N]; std::fill(ic_vals, ic_vals + N, 0.0f);
    for (auto& ic : ics) {
        ic_flags[ic.derivative - 1] = 1;
        ic_vals[ic.derivative - 1] = ic.cartesian()[axis] / zerodiffs[ic.derivative];
    }

    // Prepare LHS N-diagonal matrix and RHS vector
    Eigen::MatrixXf A(n, N + max_pivot_find); A.setZero();
    Eigen::VectorXf B(n); B.setZero();

    // Fill RHS vector with alternating sign point differences
    for (int i = 0; i < segments.size() - 1; i++) {
        for (int j = 0; j < N; j++) {
            int r = i * N + j;
            B[r] = (points[i + 1][axis] - points[i][axis])
                * ((j % 2) * -2 + 1);
        }
    }

    // Row n - bc_length - 1 for some reason needs manual filling
    A.row(n - bcs_length - 1).head(N).setConstant(1.0f);
    B[n - bcs_length - 1] = points[segments.size()][axis] - points[segments.size() - 1][axis];
    // Fill in BC differential rows
    for (int i = 0; i < bcs_length; i++) {
        int r = n - bcs_length + i;
        int d = bcs[i].derivative;
        for (int j = 0; j < N - i - 1; j++) {
            A(r, j) = Polynomial<N>::get_differential(N).coeff(j+d, d);
        }
        B[r] = bcs[i].cartesian()[axis];
        if (d == 1) { // Special case for first derivative BC because it destroys the diagonal
            B[r] -= B[n - bcs_length - 1];
        }
    }

    // IC Collapse (Creates a viable N-diagonal matrix)
    for (int i = 0; i < N; i++) {
        int r = N-1-i;
        for (int j = N-1; j >= 0; j--) {
            if (i + j < N) {
                if (!ic_flags[i + j]) {
                    A(i, r--) = get_pascal_coefficient(i, j);
                } else {
                    B[i] -= ic_vals[i + j] * get_pascal_coefficient(i, j);
                }
            } else {
                A(i, j) = get_pascal_coefficient(i, j);
            }
        }
    }

    // Use Pascal triangle to fill in the rest of the matrix
    // 12/30/2024: Why does the pascal triangle show up here?
    for (int i = N; i < n - bcs_length - 1; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = get_pascal_coefficient(i % N, j);
        }
    }

    // (Im lazy so I just added the constants (wasted 20 bytes oh nooo!!!))
    Eigen::VectorXf X(n + eff_length - 1); X.fill(0.0f);

    bool use_own_solver = true;
    if (use_own_solver) {
        // N-diagonal banded matrix solver for Ax=B with partial pivoting (Forward pass)
        for (int i = 0; i < n - 1; i++) {
            int max_magnitude_ei = i;
            int max_magnitude_ej = half;

            // Find pivot
            for (int j = 1; j <= max_pivot_find; j++) {
                int ei = i+j; int ej = half-j;
                if (ei >= n) break;

                if (fabsf(A(ei, ej)) > fabsf(A(max_magnitude_ei, max_magnitude_ej))) {
                    max_magnitude_ei = ei;
                    max_magnitude_ej = ej;
                }
            }

            // Pivot the rows
            A.row(i).tail(eff_length).swap(A.row(max_magnitude_ei).segment(max_magnitude_ej, eff_length));
            std::swap(B[i], B[max_magnitude_ei]);

            for (int j = 1; j <= half; j++) {
                int ei = i+j; int ej = half-j;
                if (ei >= n) break;

                double alpha = A(ei, ej) / A(i, half); // Use double for precision

                // Eliminate row
                A.row(ei).segment(ej, eff_length) -= alpha * A.row(i).segment(half, eff_length);
                B[ei] -= alpha * B[i];
            }
        }

        // Backsubstitute to find values
        for (int i = n - 1; i >= 0; i--) {
            X[i] = (B[i] - A.row(i).tail(eff_length-1).dot(X.segment(i+1, eff_length-1))) / A(i, half);
        }
    } else {
        Eigen::SparseMatrix<float> sparse_A(n, n);
        for (int dst_j = 0; dst_j < n; dst_j++) {
            for (int src_i = 0; src_i < N; src_i++) {
                int dst_i = dst_j + src_i - half;
                if (dst_i < 0 || dst_i >= n) continue;
                sparse_A.insert(dst_j, dst_i) = A(dst_j, src_i);
            }
        }

        Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
        solver.compute(sparse_A);

        if (solver.info() != Eigen::Success) {
            std::cerr << "Failed to decompose matrix\n";
            return;
        }

        X.head(n) = solver.solve(B);
    }

    // Because we collapsed ICs, we need to manually fill in the first segment
    auto& poly0 = axis == 0 ? segments[0].x_poly : segments[0].y_poly;
    poly0.coeffs[0] = points[0][axis];
    int fi = 0;
    for (int i = 0; i < N; i++) {
        if (ic_flags[i]) {
            poly0.coeffs[i+1] = ic_vals[i];
        } else {
            poly0.coeffs[i+1] = X[fi++];
        }
    }

    // Fill in the rest of the segments
    for (int i = 1; i < segments.size(); i++) {
        auto& poly = axis == 0 ? segments[i].x_poly : segments[i].y_poly;
        poly.coeffs[0] = points[i][axis];
        for (int j = 0; j < N; j++) {
            poly.coeffs[j+1] = X[fi++];
        }
    }
}

template <int N>
void NthDegreeSpline<N>::solve_coeffs(const std::vector<Condition>& ics, 
    const std::vector<Condition>& bcs) 
{
    segments.clear(); 
    segments.resize(points.size() - 1);

    solve_spline(0, ics, bcs);
    solve_spline(1, ics, bcs);

    int num_lengths = 32;
    lengths.resize(segments.size() * num_lengths + 1);
    lengths[0] = {0, 0};

    for (int i = 0; i < segments.size(); i++) {
        for (int j = 0; j < num_lengths; j++) {
            float t1 = (float) j / num_lengths;
            float t2 = (float) (j + 1) / num_lengths;
            int li = num_lengths * i + j;
            lengths[li+1] = {
                lengths[li].first + segments[i].length(t1, t2), 
                i + t2
            };
        }
    }
}

template <int N>
void NthDegreeSpline<N>::compute(float t, Eigen::Vector2f& res, int deriv) const {
    segments[i_helper(t)].compute(t, res, deriv);
}

template <int N>
inline Eigen::Vector2f NthDegreeSpline<N>::compute(float t, int deriv) const {
    Eigen::Vector2f res;
    compute(t, res, deriv);
    return res;
}

__DEFINE_FORWARDER(normal, Eigen::Vector2f)
__DEFINE_FORWARDER(angle, float)
__DEFINE_FORWARDER(angular_velocity, float)
__DEFINE_FORWARDER(curvature, float)

template <int N>
float NthDegreeSpline<N>::length(float t) const {
    if (t <= 0) return 0.0f;
    if (t >= maxt()) return lengths.back().first;

    int li = std::lower_bound(lengths.begin(), lengths.end(), t, [](const auto& a, const auto& b) { return a.second < b; }) - lengths.begin() - 1;
    int i = i_helper(t);
    float t0 = lengths[li].second; i_helper(t0);
    return lengths[li].first + segments[i].length(t0, t);
}

template <int N>
float NthDegreeSpline<N>::path_parametrize(float s) const {
    if (s <= 0) return 0.0f;
    if (s >= lengths.back().first) return lengths.back().second;

    int i = std::lower_bound(lengths.begin(), lengths.end(), s, [](const auto& a, const auto& b) { return a.first < b; }) - lengths.begin();
    float slope = (lengths[i-1].second - lengths[i].second) / (lengths[i-1].first - lengths[i].first); // approx du / ds
    float guess = slope * (s - lengths[i-1].first) + lengths[i-1].second;
    return guess;
}
template <int N>
void NthDegreeSpline<N>::profile_path(const ProfileParams& params) {
    constexpr size_t initial_size = 2048;
    using ConstructionPoint = ProfilePoint::ConstructionPoint;

    std::queue<ConstructionPoint> forward_pass;
    forward_pass.emplace(params.min_speed, 0.0f);

    Eigen::Vector2f deriv1;
    Eigen::Vector2f deriv2;
    float curve = curvature(0.0f);
    float scalar;
    float dkdt;

    auto predict = [&](float pp) {
        deriv1 = compute(pp, 1);
        deriv2 = compute(pp, 2);
        float inv_norm = 1.0f / (deriv1.squaredNorm() * deriv1.norm() + 1e-6f);
        float new_curve = deriv1.cross(deriv2) * inv_norm;
        dkdt = 2.0f * (new_curve < 0 ? 1 : -1) * ((new_curve - curve) / params.dt); // Why does *2 work better i dont know
        curve = new_curve;
        scalar = 1.0f / (1.0f + fabsf(curve * params.track_width * 0.5f));
    };

    predict(0.0f);
    dkdt = 0.0f;

    while (forward_pass.back().distance < length(maxt())) {
        const auto& last = forward_pass.back();

        float accel = (params.max_accel - last.speed * dkdt * params.track_width * 0.5f) * scalar;

        float new_speed = std::clamp(
            last.speed + accel * params.dt,
            params.min_speed, fmaxf(params.min_speed, params.max_speed * scalar)
        );
        float delta_dist = fminf(0.5f * (new_speed + last.speed) * params.dt, length(maxt()) - last.distance);
        float new_param = path_parametrize(last.distance + delta_dist);

        forward_pass.emplace(
            new_speed, // speed
            last.distance + delta_dist // distance
        );

        float predicted_path_param = path_parametrize(last.distance + delta_dist + new_speed * params.dt);
        predict(predicted_path_param);
    }

    curve = curvature(maxt());
    predict(maxt());
    dkdt = 0.0f;

    std::stack<ConstructionPoint> backward_pass;
    backward_pass.emplace(0.0f, length(maxt()));

    while (backward_pass.top().distance > 0.0f) {
        const auto& last = backward_pass.top();

        float accel = (params.max_accel - last.speed * dkdt * params.track_width * 0.5f) * scalar;

        float new_speed = std::clamp(
            last.speed + accel * params.dt,
            params.min_speed, fmaxf(params.min_speed, params.max_speed * scalar)
        );
        float delta_dist = fminf(0.5f * (new_speed + last.speed) * params.dt, last.distance);
        float new_param = path_parametrize(last.distance - delta_dist);

        backward_pass.emplace(
            new_speed, // speed
            last.distance - delta_dist // distance
        );

        float predicted_path_param = path_parametrize(last.distance - delta_dist - new_speed * params.dt);
        predict(predicted_path_param);
    }

    forward_pass.back().distance = length(maxt());
    backward_pass.top().distance = 0.0f;

    ConstructionPoint lf = forward_pass.front();
    ConstructionPoint lb = backward_pass.top();
    profile.clear();
    profile.reserve(initial_size);

    profile.emplace_back(
        compute(0, 0), // position
        compute(0, 1), // first derivative
        compute(0, 2), // second derivative
        0.0f, // distance
        params.min_speed, // speed
        params.max_accel, // accel
        0.0f, // path param
        0.0f // real time
    );

    while (profile.back().distance < length(maxt()) - 0.5f) {
        const auto& last = profile.back();
        float cur_distance = last.distance + params.dt * last.speed + 0.5f * last.accel * params.dt * params.dt;
        
        while ((forward_pass.size() > 1) && forward_pass.front().distance <= cur_distance) {
            lf = forward_pass.front(); forward_pass.pop();
        }
        while ((backward_pass.size() > 1) && backward_pass.top().distance <= cur_distance) {
            lb = backward_pass.top(); backward_pass.pop();
        }

        const auto& cf = forward_pass.front();
        const auto& cb = backward_pass.top();
        float slope_cf = (lf.speed * lf.speed - cf.speed * cf.speed) / (lf.distance - cf.distance);
        float slope_cb = (lb.speed * lb.speed - cb.speed * cb.speed) / (lb.distance - cb.distance);
        float cf_calc = slope_cf * (cur_distance - lf.distance) + lf.speed * lf.speed;
        float cb_calc = slope_cb * (cur_distance - lb.distance) + lb.speed * lb.speed;

        float speed = sqrtf(fmaxf(0, fminf(cf_calc, cb_calc)));
        float accel = 0.5f * (1.0f / speed);
        if (cf_calc < cb_calc) {
            accel *= slope_cf;
        } else {
            accel *= slope_cb;
        }

        float new_param = path_parametrize(cur_distance);

        profile.emplace_back(
            compute(new_param, 0), // position
            compute(new_param, 1), // first derivative
            compute(new_param, 2), // second derivative
            cur_distance, // distance
            speed, // speed
            accel, // accel
            new_param, // path param
            last.real_time + params.dt // real time
        );
    }

    profile.back().distance = length(maxt());
}

template<int N>
std::string NthDegreeSpline<N>::debug_out(void) const {
    return debug_out_precise(4);
}

template<int N>
std::string NthDegreeSpline<N>::debug_out_precise(int precision) const {
    std::stringstream result;

    result << "============== " << N << "-th Degree Spline ==============\n";
    for (int i = 0; i < segments.size(); i++) {
        char buf[4096];
        sprintf(buf, "P_{%d}\\left(t\\right) = \\left(%s,\\ %s\\right)\n", 
            i, 
            segments[i].x_poly.debug_out(precision).c_str(), 
            segments[i].y_poly.debug_out(precision).c_str()
        );
        result << buf;
    }

    result << "P\\left(t\\right) = \\left\\{";
    for (int i = 0; i < segments.size(); i++) {
        result << i << "\\le t";
        if (i != segments.size()-1) result << " < ";
        else result << "\\le ";
        result << (i + 1) << ": P_{" << i << "}\\left(t - " << i << "\\right)";
        if (i != segments.size()-1) result << ",\\ ";
    }
    result << "\\right\\}\n";

    result << "N = \\left[";
    for (int i = 0; i < points.size(); i++) {
        const auto& p = points[i];
        result << "\\left(" << p.x() << ",\\ " << p.y() << "\\right)";
        if (i != points.size()-1) result << ",\\ ";
    }
    result << "\\right]\n";
    result << "=========================================================";

    return result.str();
}

using LinearPath = NthDegreeSpline<1>;
using CubicSpline = NthDegreeSpline<3>;
using QuinticSpline = NthDegreeSpline<5>;
} // namespace pathing