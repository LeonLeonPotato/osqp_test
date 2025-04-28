#pragma once

#include "Eigen/Dense"
#include <iomanip>
#include <iostream>

namespace pathing {
static float leggauss5_nodes[5] =  {-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664};
static float leggauss5_weights[5] = {0.23692688505618942, 0.4786286704993662, 0.568888888888889, 0.4786286704993662, 0.23692688505618942};

static float leggauss20_nodes[20] = {-0.9931285991850949, -0.9639719272779138, -0.9122344282513258, -0.8391169718222188, -0.7463319064601508, -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.2277858511416451, -0.07652652113349734, 0.07652652113349734, 0.2277858511416451, 0.37370608871541955, 0.5108670019508271, 0.636053680726515, 0.7463319064601508, 0.8391169718222188, 0.9122344282513258, 0.9639719272779138, 0.9931285991850949};
static float leggauss20_weights[20] = {0.017614007139153273, 0.04060142980038622, 0.06267204833410944, 0.08327674157670467, 0.10193011981724026, 0.11819453196151825, 0.13168863844917653, 0.14209610931838187, 0.14917298647260366, 0.15275338713072578, 0.15275338713072578, 0.14917298647260366, 0.14209610931838187, 0.13168863844917653, 0.11819453196151825, 0.10193011981724026, 0.08327674157670467, 0.06267204833410944, 0.04060142980038622, 0.017614007139153273};

template <int N>
class Polynomial {
    private:
        static Eigen::MatrixXi differential_matrix;
        static void build_differential_matrix(int n);

        static int falling_factorial(int i, int n);
    public:
        static const Eigen::MatrixXi& get_differential(int n);
        static void clear_cache();

        Eigen::Vector<float, N+1> coeffs;

        Polynomial(void) { }
        Polynomial(Eigen::Vector<float, N+1>& coeffs) : coeffs(coeffs) { }

        template <typename T, typename R>
        void compute(const T& t, R res, int deriv = 0) const;

        template <typename T>
        Eigen::ArrayXf compute(const T& t, int deriv = 0) const;

        float compute(float t, int deriv = 0) const;

        template <typename... Args>
        auto operator()(Args... args) -> decltype(compute(args...)) const { return compute(args...); }

        std::string debug_out(int precision = 8) const;
};

template <int N>
class Polynomial2D {
    public:
        Polynomial<N> x_poly;
        Polynomial<N> y_poly;

        Polynomial2D(void) {}
        Polynomial2D(const Polynomial<N>& x_poly, const Polynomial<N>& y_poly) : x_poly(x_poly), y_poly(y_poly) {}

        template <typename T, typename R>
        requires (!std::is_integral<R>::value)
        void compute(const T& t, R res, int deriv = 0) const;

        template <typename T>
        requires (!std::is_integral<T>::value)
        Eigen::MatrixX2f compute(const T& t, int deriv = 0) const;

        template <typename V>
        void compute(float t, V& res, int deriv = 0) const;

        Eigen::Vector2f compute(float t, int deriv = 0) const;

        Eigen::Vector2f normal(float t) const;
        float angle(float t) const;
        float angular_velocity(float t) const;
        float curvature(float t) const;

        float length(float t) const;
        float length(float t0, float t1) const;

        template <typename... Args>
        auto operator()(Args... args) -> decltype(compute(args...)) const { return compute(args...); }
};

template <int N>
inline Eigen::MatrixXi Polynomial<N>::differential_matrix = Eigen::MatrixXi::Ones(1, 1);

template <int N>
inline void Polynomial<N>::build_differential_matrix(int n) {
    int old = differential_matrix.rows();
    differential_matrix.conservativeResize(n+1, n+1); // n+1 coefficients for n-th degree polynomial
    for (int i = old; i < n+1; i++) {
        for (int j = 0; j <= i; j++) {
            differential_matrix(i, j) = falling_factorial(i, j);
            differential_matrix(j, i) = falling_factorial(j, i);
        }
    }
}

template <int N>
inline const Eigen::MatrixXi& Polynomial<N>::get_differential(int n) {
    if (differential_matrix.rows() <= n) build_differential_matrix(n);
    return differential_matrix;
}

template <int N>
inline void Polynomial<N>::clear_cache() {
    differential_matrix.resize(1, 1);
    differential_matrix(0, 0) = 1;
}

template <int N>
inline int Polynomial<N>::falling_factorial(int i, int n) {
    if (i < n) return 0;
    int result = 1;
    for (int j = i; j > i - n; j--) result *= j;
    return result;
}

template <int N>
template <typename T, typename R>
inline void Polynomial<N>::compute(const T& t, R res, int deriv) const {
    float fact_tracker = get_differential(N).coeff(N, deriv);
    res.setConstant(coeffs.coeff(N) * fact_tracker);
    for (int i = N-1; i >= deriv; i--) {
        fact_tracker = (fact_tracker / (i + 1)) * (i - deriv + 1);
        res = (t.array() * res.array()) + coeffs.coeff(i) * fact_tracker;
    }
}

template <int N>
template <typename T>
inline Eigen::ArrayXf Polynomial<N>::compute(const T& t, int deriv) const {
    Eigen::ArrayXf y (t.size());
    compute(t, Eigen::Ref<Eigen::ArrayXf>(y), deriv);
    return y;
}

template <int N>
inline float Polynomial<N>::compute(float t, int deriv) const {
    float fact_tracker = get_differential(N).coeff(N, deriv);
    float result = coeffs.coeff(N) * fact_tracker;
    for (int i = N-1; i >= deriv; i--) {
        fact_tracker = (fact_tracker / (i + 1)) * (i - deriv + 1);
        result = result * t + coeffs.coeff(i) * fact_tracker;
    }
    return result;
}

template <int N>
inline std::string Polynomial<N>::debug_out(int precision) const {
    std::string result = "";
    for (int i = 0; i <= N; i++) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(precision);
        if (i == 0) out << coeffs(i);
        else out << fabs(coeffs(i));
        std::string coeff = out.str();

        std::string term = "t^{" + std::to_string(i) + "}";
        if (i == 1) term = "t";
        if (i == 0) term = "";

        std::string nxt = " + ";
        if (i == N) nxt = "";
        else if (coeffs(i+1) < 0) nxt = " - ";

        result += coeff + term + nxt;
    }
    return result;
}

template <int N>
template <typename T, typename R>
requires (!std::is_integral<R>::value)
inline void Polynomial2D<N>::compute(const T& t, R res, int deriv) const {
    x_poly.compute(t, res.col(0), deriv);
    y_poly.compute(t, res.col(1), deriv);
}

template <int N>
template <typename T>
requires (!std::is_integral<T>::value)
inline Eigen::MatrixX2f Polynomial2D<N>::compute(const T& t, int deriv) const {
    Eigen::MatrixX2f x(t.size(), 2);
    compute(t, x, deriv);
    return x;
}

template <int N>
template <typename V>
inline void Polynomial2D<N>::compute(float t, V& res, int deriv) const {
    float fact_tracker = x_poly.get_differential(N).coeff(N, deriv);
    res.coeffRef(0) = x_poly.coeffs.coeff(N) * fact_tracker;
    res.coeffRef(1) = y_poly.coeffs.coeff(N) * fact_tracker;
    for (int i = N-1; i >= deriv; i--) {
        fact_tracker = (fact_tracker / (i + 1)) * (i - deriv + 1);
        res.coeffRef(0) = res.coeff(0) * t + x_poly.coeffs.coeff(i) * fact_tracker;
        res.coeffRef(1) = res.coeff(1) * t + y_poly.coeffs.coeff(i) * fact_tracker;
    }
}

template <int N>
inline Eigen::Vector2f Polynomial2D<N>::compute(float t, int deriv) const {
    Eigen::Vector2f x;
    compute(t, x, deriv);
    return x;
}

template <int N>
inline Eigen::Vector2f Polynomial2D<N>::normal(float t) const {
    Eigen::Vector2f d = compute(t, 1);
    return Eigen::Vector2f(-d(1), d(0));
}

template <int N>
inline float Polynomial2D<N>::angle(float t) const {
    Eigen::Vector2f d = compute(t, 1);
    return atan2f(d(0), d(1));
}

template <int N>
inline float Polynomial2D<N>::angular_velocity(float t) const {
    Eigen::Vector2f d1 = compute(t, 1);
    Eigen::Vector2f d2 = compute(t, 2);
    return (d1(0) * d2(1) - d1(1) * d2(0)) / (d1.dot(d1) + 1e-6);
}

template <int N>
inline float Polynomial2D<N>::curvature(float t) const {
    const Eigen::Vector2f d1 = compute(t, 1);
    const Eigen::Vector2f d2 = compute(t, 2);
    return (d1(0) * d2(1) - d1(1) * d2(0)) / ((d1(0) * d1(0) + d1(1) * d1(1)) * 
        sqrtf(d1(0) * d1(0) + d1(1) * d1(1)) + 1e-6);
}

template <int N>
inline float Polynomial2D<N>::length(float t) const {
    if (t == 0) return 0;

    float length = 0.0f;
    for (int i = 0; i < 5; i++) {
        float x = t * (leggauss5_nodes[i] * 0.5f + 0.5f);
        length += (t * 0.5f) * compute(x, 1).norm() * leggauss5_weights[i];
    }

    return length;
}

template <int N>
inline float Polynomial2D<N>::length(float t0, float t1) const {
    float length = 0.0f;
    for (int i = 0; i < 5; i++) {
        float x = t0 + (t1 - t0) * (leggauss5_nodes[i] * 0.5f + 0.5f);
        length += ((t1 - t0) * 0.5f) * compute(x, 1).norm() * leggauss5_weights[i];
    }

    return length;
}
}