#include "testing/testing.h"
#include <cmath>
#include <cstdio>
#include "actuator.h"
#include "hpipm/hpipm_s_ocp_qp_ipm.h"
#include "hpipm/hpipm_s_ocp_qp_sol.h"
#include "localization.h"
#include "mpclib/models.h"
#include "mpclib/ocp.h"
#include "pros/rtos.h"
#include "pros/rtos.hpp"
#include "Eigen/Geometry"
#include <algorithm>
#include "autodiff/forward/dual.hpp"

using namespace mpclib;

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
        float derivative = prev_time_ == -1 ? 0 : (error - prev_error_) / dt;
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

void test_model() {
    DifferentialDriveModel::Params params;
    params.dt = 1.0f;
    params.width = 34.5f;
    params.max_speed = 150.0f;
    params.acceleration_constant = 2.4f;

    DifferentialDriveModel model(params);

    // Inference test
    Vec x(5);
    x << 0, 0, 0, 0, 0;
    Vec u(2);
    u << 150, 0;
    auto result = model.infer(x, u);
    printf("Result: %f %f %f %f %f\n", result[0], result[1], result[2], result[3], result[4]);

    // General constraints test
    std::cout << "General Constraints State Matrix:\n";
    std::cout << model.general_constraints_state_matrix() << std::endl;
    std::cout << "General Constraints Action Matrix:\n";
    std::cout << model.general_constraints_action_matrix() << std::endl;

    // Getting parameters
    auto model_params = model.params();
    model_params.acceleration_constant = 3.0f;
    model_params.dt = 0.5f;
    model.set_params(model_params);

    // Inference test after changing parameters
    result = model.infer(x, u);
    printf("Result after changing parameters: %f %f %f %f %f\n", result[0], result[1], result[2], result[3], result[4]);

    // General constraints test after changing parameters
    std::cout << "General Constraints State Matrix after changing parameters:\n";
    std::cout << model.general_constraints_state_matrix() << std::endl;
    std::cout << "General Constraints Action Matrix after changing parameters:\n";
    std::cout << model.general_constraints_action_matrix() << std::endl << std::endl;
}

void test_ocp_qp() {
    DifferentialDriveModel::Params model_params;
    model_params.dt = 0.05f;
    model_params.width = 34.5f;
    model_params.max_speed = 160.0f;
    model_params.acceleration_constant = 0.401f;
    DifferentialDriveModel model(model_params);

    DifferentialDriveModel::Params simulation_params;
    simulation_params.dt = 0.02f;
    simulation_params.width = 34.5f;
    simulation_params.max_speed = 160.0f;
    simulation_params.acceleration_constant = 0.401f;
    DifferentialDriveModel simulator(simulation_params);

    float time_target = 5.0; // ms
    OCPParams ocp_params;
    ocp_params.N = 20;
    ocp_params.Q = (Eigen::Vector<float, 5> {1, 1, 0, 0.00, 0.00}).asDiagonal();
    ocp_params.Qf = ocp_params.Q * 1.5;
    ocp_params.R = Mat::Identity(2, 2) * 0.01;
    ocp_params.Rf = Mat::Identity(2, 2) * 0.04;
    ocp_params.warm_start_level = OCPParams::WarmStartLevel::NONE;
    ocp_params.iterations = (int) std::round(time_target / (2.500000e-02 * ocp_params.N));

    Vec x0(5); x0 << 0, 0, 0.1, 1, 0;
    Vec u0(2); u0 << 1, 0;
    Vec x_target(5); x_target << -100, 100, 0, 0, 0;

    OCPQP ocpqp(model, ocp_params);// [2.920, 3.306] [-1.540, -0.363]
    ocpqp.set_initial_state(x0);
    ocpqp.relinearize(x0, u0);
    ocpqp.set_target_state(x_target);

    // Simulate the model with the solution
    std::vector<Vec> positions;
    std::vector<Vec> actions;
    positions.push_back(x0.head(2));
    for (int i = 0; i < 1200; i++) {
        if ((i+1) % 300 == 0) {
            float tx = x_target(0);
            float ty = x_target(1);
            x_target.head(2) << -ty, tx;
            ocpqp.set_target_state(x_target);
        }

        auto t1 = pros::micros();
        int status = ocpqp.solve(false);
        auto t2 = pros::micros();

        if (status == 3) break;

        s_ocp_qp_sol_get_u(0, &ocpqp.qp_sol, u0.data());
        u0 * simulator.params().max_speed / model.params().max_speed;
        x0 = simulator.infer(x0, u0 + Vec::Random(u0.size()) * 20);

        ocpqp.set_initial_state(x0 + Vec::Random(x0.size()) * 0.1f);
        
        int stage = 5;
        Vec u_nom(2); s_ocp_qp_sol_get_u(stage+1, &ocpqp.qp_sol, u_nom.data());
        Vec x_nom(5); s_ocp_qp_sol_get_x(stage, &ocpqp.qp_sol, x_nom.data());
        if (stage == 0) x_nom = x0;
        ocpqp.relinearize(x_nom, u0);

        // push down x
        Vec x_buffer(5);
        s_ocp_qp_sol_set_x(0, x0.data(), &ocpqp.qp_sol);
        for (int j = 1; j < ocp_params.N; j++) {
            s_ocp_qp_sol_get_x(j+1, &ocpqp.qp_sol, x_buffer.data());
            s_ocp_qp_sol_set_x(j, x_buffer.data(), &ocpqp.qp_sol);
        }

        // push down u
        Vec u_buffer(2);
        for (int j = 0; j < ocp_params.N - 1; j++) {
            s_ocp_qp_sol_get_u(j+1, &ocpqp.qp_sol, u_buffer.data());
            s_ocp_qp_sol_set_u(j, u_buffer.data(), &ocpqp.qp_sol);
        }

        auto t3 = pros::micros();
        printf("Step %d Solve time: %lld us         ", i, t2 - t1);
        printf("Action: [%.2f, %.2f] State: [%.3f, %.3f, %.3f, %.3f, %.3f]      ", u0[0], u0[1], x0[0], x0[1], x0[2], x0[3], x0[4]);
        printf("Total iteration time: %lld us\n", t3 - t1);

        actions.push_back(u0);
        if (i % 3 == 0)
            positions.push_back(x0.head(2));
    }

    int rows = actions.size();
    int cols = actions[0].size();
    Eigen::MatrixXf mat(rows, cols);

    for (int i = 0; i < rows; ++i) mat.row(i) = actions[i];

    mat = mat.cwiseAbs();
    printf("Mean action: [%.3f, %.3f]\n", mat.col(0).mean(), mat.col(1).mean());

    std::cout << "x_t = [";
    for (int i = 0; i < positions.size(); i++) {
        auto& v = positions[i];
        printf("(%.3f,%.3f)", v.x(), v.y());
        if (i != positions.size() - 1) printf(",");
    }
    printf("]\n");
}

static Vec get_target_at_time(float time) {
    static Mat x_coeffs(4, 4);
    static Mat y_coeffs(4, 4);
    static std::vector<float> arc_lengths;
    static bool initialized = false;

    static float nodes[20] = {-0.9931286 , -0.96397193, -0.91223443, -0.83911697, -0.74633191, -0.63605368, -0.510867  , -0.37370609, -0.22778585, -0.07652652,         0.07652652,  0.22778585,  0.37370609,  0.510867  ,  0.63605368,         0.74633191,  0.83911697,  0.91223443,  0.96397193,  0.9931286 };
    static float weights[20] = {0.01761401, 0.04060143, 0.06267205, 0.08327674, 0.10193012,
       0.11819453, 0.13168864, 0.14209611, 0.14917299, 0.15275339,
       0.15275339, 0.14917299, 0.14209611, 0.13168864, 0.11819453,
       0.10193012, 0.08327674, 0.06267205, 0.04060143, 0.01761401};
    
    auto compute = [](auto t) {
        int interval = static_cast<int>(std::floor(static_cast<float>(t)));
        interval = std::clamp(interval, 0, x_coeffs.rows() - 1);
        auto dt = t - interval;
        if (dt < 0) dt = decltype(dt)(0);
        if (dt > 1) dt = decltype(dt)(1);

        auto x = x_coeffs(interval, 0)
                + x_coeffs(interval, 1) * dt
                + x_coeffs(interval, 2) * dt * dt
                + x_coeffs(interval, 3) * dt * dt * dt;

        auto y = y_coeffs(interval, 0)
                + y_coeffs(interval, 1) * dt
                + y_coeffs(interval, 2) * dt * dt
                + y_coeffs(interval, 3) * dt * dt * dt;

        return Eigen::Vector<float, 2> {x, y};
    };

    auto derivative = [](auto t) {
        int interval = static_cast<int>(std::floor(static_cast<float>(t)));
        interval = std::clamp(interval, 0, x_coeffs.rows() - 1);
        auto dt = t - interval;
        dt = std::clamp(dt, 0.0f, 1.0f);

        auto dx = x_coeffs(interval, 1)
                + 2 * x_coeffs(interval, 2) * dt
                + 3 * x_coeffs(interval, 3) * dt * dt;

        auto dy = y_coeffs(interval, 1)
                + 2 * y_coeffs(interval, 2) * dt
                + 3 * y_coeffs(interval, 3) * dt * dt;

        return Eigen::Vector<float, 2> {dx, dy};
    };

    auto arc_length_to_parameter = [&compute, &derivative](float s) {
        float guess = (float) (std::lower_bound(arc_lengths.begin(), arc_lengths.end(), s) - arc_lengths.begin());
        guess = (guess / arc_lengths.size()) * x_coeffs.rows();

        float check = 0;
        for (int i = 0; i < 20; i++) {
            float f = derivative(guess * (1 + nodes[i]) / 2).norm();
            check += weights[i] * f;
        }
        check *= (guess * 0.5f);

        auto deriv = derivative(guess).norm();
        if (deriv != 0)
            guess = guess - ((check - s) / deriv);

        return guess;
    };

    auto arc_length = [&derivative](float t) {
        float check = 0;
        for (int i = 0; i < 20; i++) {
            float f = derivative(t * (1 + nodes[i]) / 2).norm();
            check += weights[i] * f;
        }
        check *= (t * 0.5f);
        return check;
    };

    if (!initialized) {
        x_coeffs << 
            -99.8,     0.3214285714285552,0.0,49.178571428571445,
            -50.3,   147.8571428571429,147.53571428571425,-132.09285714285713,
            113.0,   46.65,-248.74285714285716,136.09285714285716,
            47.0,    -42.55714285714285,159.53571428571428,-53.17857142857143;

        y_coeffs <<
            -124.5,  278.203571, 2.84217e-14, -98.103571,
            55.6,   -16.107143, -294.310714, 156.417857,
            -98.4, -135.475000, 174.942857,  -22.067857,
            -81.0,  148.207143, 108.739286,  -36.246429;

        int segments = 1 << 8;
        arc_lengths.reserve(1 + x_coeffs.rows() * segments);
        arc_lengths.push_back(0);

        Vec last = compute(0);
        for (int i = 0; i < x_coeffs.rows() * segments; i++) {
            float t = i / (float)segments;
            Vec cur = compute(t);
            float dx = cur[0] - last[0];
            float dy = cur[1] - last[1];
            float length = sqrtf(dx * dx + dy * dy);
            arc_lengths.push_back(arc_lengths.back() + length);
            last = cur;
        }

        initialized = true;
    }

    float speed = 75.0f;
    float total_s = arc_length(static_cast<float>(x_coeffs.rows()));
    time = modfix(time, total_s / speed + 5);
    time -= 5;
    float s = time * speed;
    time = arc_length_to_parameter(s);
    time = std::clamp(time, 0.0f, static_cast<float>(x_coeffs.rows()));

    return (Vec(5) << compute(time), 0, 0, 0).finished(); // or Vector2f{x, y} depending on what `Vec` is
}

void test_in_sim() {
    DifferentialDriveModel::Params model_params;
    float gain = 13.6; float kf = 0.7;
    model_params.dt = 0.1f;
    model_params.width = 35.87f;
    model_params.max_speed = (gain - kf) * 12.0f;
    model_params.acceleration_constant = 0.1404f;
    DifferentialDriveModel model(model_params);

    float time_target = 7.5; // ms
    OCPParams ocp_params;
    ocp_params.N = 30;
    ocp_params.Q = (Eigen::Vector<float, 5> {1, 1, 0, 0.000, 0.00}).asDiagonal();
    ocp_params.Qf = (Eigen::Vector<float, 5> {10, 10, 0, 0.000, 0.00}).asDiagonal();
    ocp_params.R = Mat::Identity(2, 2) * 0.000;
    ocp_params.Rf = Mat::Identity(2, 2) * 0.00;
    ocp_params.warm_start_level = OCPParams::WarmStartLevel::STATE_AND_INPUT;
    ocp_params.iterations = (int) std::round(time_target / (2.500000e-02 * ocp_params.N));
    OCPQP ocpqp(model, ocp_params);

    SimulatedActuator actuator;
    SimulatedLocalizer localizer;
    MotorController left_controller(1.2, 0.3, 0.00, 1 / gain, kf, 0.025);
    MotorController right_controller(1.2, 0.3, 0.00, 1 / gain, kf, 0.025);

    std::vector<Vec> targets;
    targets.reserve(ocp_params.N);

    Vec x_nom(5);
    Vec u_nom(2); u_nom << 0, 0;

    while (true) {
        uint32_t loop_start_time = pros::millis();
        x_nom = localizer.get_state();

        if (targets.size() != 0) {
            float error = sqrtf(powf(x_nom[0] - targets[0][0], 2) + powf(x_nom[1] - targets[0][1], 2));
            printf("Error: %.3f     Target: [%.3f, %.3f]\n", error, targets[0][0], targets[0][1]);
            printf("[RENDER] %.4f %.4f %f\n", targets[0][0], targets[0][1], 10.0);
        }

        targets.clear();
        float cur_time = pros::micros() * 1e-6f;
        for (int i = 1; i <= ocp_params.N; i++) {
            targets.push_back(get_target_at_time(cur_time + model_params.dt * i));
        }

        ocpqp.set_initial_state(x_nom);
        ocpqp.relinearize(x_nom, u_nom);
        ocpqp.set_target_state(targets);

        int status = ocpqp.solve(false);
        if (status == 3) break;

        std::vector<Vec> pred_states(ocp_params.N); 
        for (int i = 0; i < pred_states.size(); i++) {
            pred_states[i].resize(model.state_size());
            s_ocp_qp_sol_get_x(i+1, &ocpqp.qp_sol, pred_states[i].data());
        }

        std::vector<Vec> pred_actions(ocp_params.N); 
        for (int i = 0; i < pred_actions.size(); i++) {
            pred_actions[i].resize(model.action_size());
            s_ocp_qp_sol_get_u(i, &ocpqp.qp_sol, pred_actions[i].data());
        }

        // int lin_stage = ocp_params.N / 2;
        int lin_stage = 0;
        ocpqp.relinearize(pred_states[lin_stage], pred_actions[lin_stage]);
        // ocpqp.relinearize(pred_states, pred_actions);

        Vec nxt_x(5);
        nxt_x = model.infer(x_nom, pred_actions[0]);
        nxt_x = 0.75 * nxt_x + 0.25 * model.infer(nxt_x, pred_actions[1]);

        actuator.volt(left_controller.calculate_voltage(x_nom[3], nxt_x[3], u_nom[0]), 
                    right_controller.calculate_voltage(x_nom[4], nxt_x[4], u_nom[1]));

        char buff[256];
        int pos = sprintf(buff, "[LINE]");
        for (int i = 0; i < pred_states.size(); i += pred_states.size() / 10) {
            pos += sprintf(buff + pos, " %.2f %.2f", pred_states[i].x(), pred_states[i].y());
        }
        sprintf(buff + pos, "\n");
        std::cout << buff;

        u_nom = pred_actions[0];

        pros::c::task_delay_until(&loop_start_time, 10);
    }
}

void find_center() {
    SimulatedLocalizer localizer;
    SimulatedActuator actuator;
    actuator.volt(-12, 12);
    while (true) {
        auto x = localizer.get_state();
        printf("x: [%.3f, %.3f, %.3f]\n", x[0], x[1], x[2]);
        pros::c::task_delay(10);
    }
}