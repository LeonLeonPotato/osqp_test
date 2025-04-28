#pragma once

#include "actuator.h"
#include "localization.h"
#include <iostream>
#include <sstream>
#include <vector>

using namespace mpclib;

namespace testing {

void driver();
void model(); 
void ocp_qp();
void simulation();
void find_center();
void motor_constants();
void test_motor_constants();
void spline();

extern SimulatedActuator* actuator;
extern SimulatedLocalizer* localizer;

/**
 * @brief Prints a vector of vectors in a formatted manner.
 * 
 * @details
 * This function takes a vector of vectors (`std::vector<Vec>`) and prints it to the console
 * in a human-readable format. Each inner vector is enclosed in parentheses (`()`), and the
 * entire collection is enclosed in square brackets (`[]`). Elements within each vector are
 * separated by commas, and the vectors themselves are also separated by commas. The output
 * precision and formatting can be customized.
 * 
 * Example:
 * ```
 * Input: vecs = {{1.234, 2.345}, {3.456, 4.567}}, precision = 2, fixed = true, name = "Matrix"
 * Output: Matrix = [(1.23, 2.35), (3.46, 4.57)]
 * ```
 * 
 * @param vecs A vector of vectors to be printed. Each inner vector represents a row or group of values.
 * @param precision The number of decimal places to display for each value (default: 3).
 * @param fixed Whether to use fixed-point notation for the values (default: true).
 * @param name A label to display before the formatted output (default: "P").
 */
static void print_vectors(const std::vector<Vec>& vecs, 
    int precision = 3, bool fixed = true,
    const std::string& name = "P") 
{
    std::ostringstream oss;
    oss << std::setprecision(precision);
    if (fixed) oss << std::fixed;

    oss << name << " = [";
    for (size_t i = 0; i < vecs.size(); ++i) {
        oss << "(";
        for (size_t j = 0; j < vecs[i].size(); ++j) {
            oss << vecs[i][j];
            if (j != vecs[i].size() - 1) oss << ", ";
        }
        oss << ")";
        if (i != vecs.size() - 1) oss << ", ";
    }
    oss << "]";

    std::cout << oss.str() << std::endl;
}

/**
 * @brief Prints a vector of floats in a formatted manner.
 * 
 * @details
 * This function takes a vector of floats (`std::vector<float>`) and prints it to the console
 * in a human-readable format. The vector is enclosed in square brackets (`[]`), and each
 * element is separated by a comma. The output precision and formatting can be customized.
 * 
 * Example:
 * ```
 * Input: vecs = {1.2345, 2.3456, 3.4567}, precision = 2, fixed = true, name = "Vector"
 * Output: Vector = [1.23, 2.35, 3.46]
 * ```
 * 
 * @param vecs A vector of floats to be printed.
 * @param precision The number of decimal places to display (default: 3).
 * @param fixed Whether to use fixed-point notation (default: true).
 * @param name A label to display before the vector (default: "P").
 */
static void print_vector(const std::vector<float>& vecs, 
    int precision = 3, bool fixed = true,
    const std::string& name = "P") 
{
    std::ostringstream oss;
    oss << std::setprecision(precision);
    if (fixed) oss << std::fixed;

    oss << name << " = [";
    for (size_t i = 0; i < vecs.size(); ++i) {
        oss << vecs[i];
        if (i != vecs.size() - 1) oss << ", ";
    }
    oss << "]";

    std::cout << oss.str() << std::endl;
}

class PID {
    public:
        float kp_;
        float ki_;
        float kd_;
        float prev_error_;
        float integral_;
        long long prev_time_;
    
        PID(float kp, float ki, float kd) : kp_(kp), ki_(ki), kd_(kd), prev_error_(0), integral_(0), prev_time_(-1) {}
    
        float compute(float setpoint, float measured_value) {
            float dt;
            if (prev_time_ == -1) {
                dt = 0;
            } else {
                dt = (pros::micros() - prev_time_) * 1e-6f;
            }
    
            float error = setpoint - measured_value;
            integral_ += error * dt;
            float derivative = (prev_time_ == -1) ? 0 : ((error - prev_error_) / dt);
            prev_error_ = error;
            prev_time_ = pros::micros();
    
            return kp_ * error + ki_ * integral_ + kd_ * derivative;
        }
};

class MotorController {
    PID pid;
    float kv;
    float kf;
    float ka;

public:
    MotorController(float kp, float ki, float kd, float kv, float kf, float ka)
        : pid(kp, ki, kd), kv(kv), kf(kf), ka(ka) {}

    float calculate_voltage(float current_speed, float target_speed, float target_acceleration = 0) {
        float feedforward = kv * target_speed + ka * target_acceleration + kf * (target_speed > 0 ? 1 : -1) * (target_speed != 0);
        float feedback = pid.compute(target_speed, current_speed);
        return feedforward + feedback;
    }
};
}