#include "testing/testing.h"
#include "n_degree_spline.h"
#include <cmath>
#include <cstdio>
#include "actuator.h"
#include "hpipm/hpipm_common.h"
#include "hpipm/hpipm_s_ocp_qp_sol.h"
#include "localization.h"
#include "mpclib/models.h"
#include "mpclib/ocp.h"
#include "pros/rtos.h"
#include <vector>

using namespace mpclib;

static SimulatedActuator actuator;
static SimulatedLocalizer localizer;

/**
 * @brief Prints a list of vectors in a formatted manner.
 * 
 * @details
 * This function takes a list of vectors (`std::vector<Vec>`) and prints them to the console
 * in a human-readable format. Each vector is represented as a tuple of its elements, and
 * the entire list is enclosed in square brackets (`[]`). For example:
 * 
 * Input: A list of vectors `[(1.0, 2.0), (3.0, 4.0)]`
 * Output: `P = [(1.000, 2.000), (3.000, 4.000)]`
 * 
 * The function uses a static buffer to construct the formatted string and ensures that
 * the buffer does not overflow.
 * 
 * @param vecs A list of vectors (`std::vector<Vec>`) to be printed. Each `Vec` is assumed
 *             to be a container (e.g., Eigen vector) with a `size()` method and index-based
 *             access to its elements.
 * 
 * @note
 * - The buffer size is fixed at 4096 bytes. If the formatted string exceeds this size,
 *   the output may be truncated.
 * - The function is designed for debugging and visualization purposes.
 */
static void print_vectors(const std::vector<Vec>& vecs) {
    static constexpr int vec_printing_buff_size = 1 << 12;
    static char* vec_printing_buff = new char[vec_printing_buff_size];

    int pos = snprintf(vec_printing_buff, vec_printing_buff_size, "P = [");
    for (int i = 0; i < vecs.size(); ++i) {
        pos += snprintf(vec_printing_buff + pos, vec_printing_buff_size - pos, "(");
        for (int j = 0; j < vecs[i].size(); ++j) {
            pos += snprintf(vec_printing_buff + pos, vec_printing_buff_size - pos, "%.3f", vecs[i][j]);
            if (j != vecs[i].size() - 1) {
                pos += snprintf(vec_printing_buff + pos, vec_printing_buff_size - pos, ", ");
            }
        }
        pos += snprintf(vec_printing_buff + pos, vec_printing_buff_size - pos, ")");
        if (i != vecs.size() - 1) pos += snprintf(vec_printing_buff + pos, vec_printing_buff_size - pos, ", ");
    }
    pos += snprintf(vec_printing_buff + pos, vec_printing_buff_size - pos, "]");
    vec_printing_buff[pos] = '\0';
    printf("%s\n", vec_printing_buff);
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

void test_in_sim() {
    float gain = 13.6; float kf = 0.7;

    DifferentialDriveModel::Params model_params;
    model_params.dt = 0.04f;
    model_params.width = 33.87f;
    model_params.max_speed = (gain - kf) * 12.0f;
    model_params.acceleration_constant = 5.39002f;
    DifferentialDriveModel model(model_params);

    float time_target = 10; // ms
    float loop_target = 15; // ms
    OCPParams ocp_params;
    ocp_params.N = 30;
    ocp_params.Q1 = (Eigen::Vector<float, 5> {10, 10, 1000, 0.007, 0.007}).asDiagonal();
    ocp_params.Q = (Eigen::Vector<float, 5> {1, 1, 500, 0.002, 0.002}).asDiagonal();
    ocp_params.Qf = (Eigen::Vector<float, 5> {5, 5, 0, 0.02, 0.02}).asDiagonal();
    ocp_params.R0 = Mat::Identity(2, 2) * 0.017;
    ocp_params.R = Mat::Identity(2, 2) * 0.002;
    ocp_params.Rf = Mat::Identity(2, 2) * 0.002;
    ocp_params.warm_start_level = OCPParams::WarmStartLevel::STATE;
    ocp_params.iterations = (int) std::round(time_target / (2.500000e-02 * ocp_params.N));
    OCPQP ocpqp(model, ocp_params);

    MotorController left_controller(0.4, 0.0, 0.005, 0.0734292, 0.71082, 0.0191517541031);
    MotorController right_controller(0.4, 0.0, 0.005, 0.0734292, 0.71082, 0.0191517541031);

    pathing::CubicSpline spline {
        {{0, 0}, {100, 100}, {50, 20}}
    };
    spline.solve_coeffs(pathing::CubicSpline::natural_conditions, pathing::CubicSpline::natural_conditions);

    std::vector<Vec> targets;
    targets.reserve(ocp_params.N);

    Vec x_nom(5); x_nom << 0, 0, 0, 0, 0;
    Vec u_nom(2); u_nom << 0, 0;

    while (true) {
        uint32_t loop_start_time = pros::millis();
        uint64_t loop_start_time_hf = pros::c::micros();
        x_nom = localizer.get_state();

        if (targets.size() != 0) {
            float error = sqrtf(powf(x_nom[0] - targets[0][0], 2) + powf(x_nom[1] - targets[0][1], 2));
            printf("Error: %.3f     Target: [%.3f, %.3f, %.3f]\n", error, targets[0][0], targets[0][1], targets[0][2]);
            printf("[RENDER] %.4f %.4f %f\n", targets[0][0], targets[0][1], 20.0);
        }

        targets.clear();
        float cur_time = pros::micros() * 1e-6f;
        for (int i = 1; i <= ocp_params.N; i++) {
            Vec target(5);
            float t = cur_time + model_params.dt*i + time_target*1e-3;
            t = modfix(t*0.1, spline.maxt());
            target << spline.compute(t), 0, 0, 0;
            targets.push_back(target);
        }
        
        x_nom = model.infer(x_nom, u_nom, time_target * 1e-3);

        ocpqp.set_initial_state(x_nom);
        ocpqp.relinearize(x_nom, u_nom);
        ocpqp.set_target_state(targets);

        int status = ocpqp.solve(false);
        if (status == hpipm_status::NAN_SOL) break;

        std::vector<Vec> pred_states = ocpqp.get_solution_states();
        std::vector<Vec> pred_actions = ocpqp.get_solution_actions();

        actuator.volt(left_controller.calculate_voltage(x_nom[3], pred_states[0][3], u_nom[0]), 
                    right_controller.calculate_voltage(x_nom[4], pred_states[0][4], u_nom[1]));

        char buff[256];
        int pos = sprintf(buff, "[LINE]");
        for (int i = 0; i < pred_states.size(); i += pred_states.size() / 10) {
            pos += sprintf(buff + pos, " %.2f %.2f", pred_states[i].x(), pred_states[i].y());
        }
        sprintf(buff + pos, "\n");
        std::cout << buff;

        u_nom = pred_actions[1];

        if ((pros::micros() - loop_start_time_hf) * 1e-3f > loop_target) {
            printf("WARNING: Loop time exceeded %d ms\n", (int) loop_target);
        }

        pros::c::task_delay_until(&loop_start_time, loop_target);
    }
}

void find_center() {
    actuator.volt(-12, 12);
    while (true) {
        auto x = localizer.get_state();
        printf("x: [%.3f, %.3f, %.3f]\n", x[0], x[1], x[2]);
        pros::c::task_delay(10);
    }
}

void motor_constants() {
    int stage = 8;
    actuator.volt(stage / 10.0f * 12, stage / 10.0f * 12);
    std::vector<Vec> positions;

    auto start = pros::micros();
    auto last_t = start;
    float last_x = localizer.get_state()[0];
    pros::delay(20);
    for (int i = 0; i < 50; i++) {
        auto x = localizer.x();
        auto t = (pros::micros() - start) * 1e-6f;
        auto dt = (pros::micros() - last_t) * 1e-6f;
        auto dx = x - last_x;
        last_x = x;
        last_t = pros::micros();
        positions.push_back((Vec(2) << t, (dx/dt)).finished());
        pros::delay(20);
    }

    print_vectors(positions);

    while (true) pros::delay(50);
}

void test_motor_constants() {
    MotorController left_controller(0.4, 0.0, 0.004, 0.0734292, 0.71082, 0.0191517541031);
    MotorController right_controller(0.4, 0.0, 0.004, 0.0734292, 0.71082, 0.0191517541031);

    float test_velo = 100;

    std::vector<Vec> positions;
    positions.reserve(100);
    for (int i = 0; i < 100;) {
        auto t = i * 0.1f;
        float vl = localizer.vl();
        float vr = localizer.vr();
        actuator.volt(
            left_controller.calculate_voltage(vl, test_velo),
            right_controller.calculate_voltage(vr, test_velo)
        );
        
        if ((fabsf(vl) + fabsf(vr)) < 2.5f) {
            pros::delay(10);
            continue;
        } else i++;

        positions.push_back((Vec(2) << t, vl).finished());
        pros::delay(10);
    }

    print_vectors(positions);

    while (true) pros::delay(50);
}