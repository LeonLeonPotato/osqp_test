#include "testing/test.h"
#include "mpclib/models.h"
#include "mpclib/ocp.h"
#include "testing/n_degree_spline.h"

using namespace testing;

void testing::simulation() {
    float gain = 13.6; float kf = 0.7;

    DifferentialDriveModel::Params model_params;
    model_params.dt = 0.025f;
    model_params.width = 33.87f;
    model_params.max_speed = (gain - kf) * 12.0f;
    model_params.acceleration_constant = 8.4f;
    DifferentialDriveModel model(model_params);

    float time_target = 7; // ms
    float loop_target = 10; // ms
    OCPParams ocp_params;
    ocp_params.N = 40;
    ocp_params.Q1 = (Eigen::Vector<float, 5> {1, 1, 0, 0.1, 0.1}).asDiagonal();
    ocp_params.Q = (Eigen::Vector<float, 5> {10, 10, 0, 0.00, 0.00}).asDiagonal();
    ocp_params.Qf = (Eigen::Vector<float, 5> {10, 10, 0, 0.00, 0.00}).asDiagonal();
    ocp_params.R0 = Mat::Identity(2, 2) * 0.001;
    ocp_params.R = Mat::Identity(2, 2) * 0.001;
    ocp_params.Rf = Mat::Identity(2, 2) * 0.001;
    ocp_params.warm_start_level = OCPParams::WarmStartLevel::STATE;
    ocp_params.iterations = (int) std::round(time_target / (2.500000e-02 * ocp_params.N));
    OCPQP ocpqp(model, ocp_params);

    MotorController left_controller(0.4, 0.0, 0.000, 0.0734292, 0.71082, 0.0291517541031);
    MotorController right_controller(0.4, 0.0, 0.000, 0.0734292, 0.71082, 0.0291517541031);

    // pathing::CubicSpline spline {{
    //     {0, 0}, 
    //     {0, 100}, 
    //     {110, 0}, 
    //     {50, -50}, 
    //     {-100, -100}, 
    //     {-110, 0}, 
    //     {10, 0}
    // }};
    pathing::CubicSpline spline {
        {{0, 0}, {100, 100}, {110, 0}, {-50, -50}, {-100, -100}, {-110, 0}, {0, 0}}
    };
    spline.solve_coeffs(
        {pathing::Condition{1, M_PI/2, 250}}, 
        {pathing::Condition{1, M_PI/2, 250}}
    );
    auto profile_start = pros::micros();
    const pathing::ProfileParams profile_params {
        .max_speed = (gain - kf) * 12.0f,
        .max_accel = 200.0f,
        .track_width = 33.87f,
        .dt = 0.01f
    };
    spline.profile_path(profile_params);
    auto profile_end = pros::micros();
    printf("Profile time: %f ms\n", (profile_end - profile_start) / 1000.0f);

    std::vector<Vec> targets;
    targets.reserve(ocp_params.N);

    Vec x_nom(5); x_nom << 0, 0, 0, 0, 0;
    Vec u_nom(2); u_nom << 0, 0;

    while (true) {
        uint32_t loop_start_time = pros::millis();
        uint64_t loop_start_time_hf = pros::c::micros();
        x_nom = localizer->get_state();

        float error = 0.0f;
        if (targets.size() != 0) {
            error = sqrtf(powf(x_nom[0] - targets[0][0], 2) + powf(x_nom[1] - targets[0][1], 2));
            printf("Error: %.3f     Target: [%.3f, %.3f, %.3f | %.2f, %.2f]     Curernt speeds: %.2f, %.2f\n", 
                error, 
                targets[0][0], targets[0][1], targets[0][2], targets[0][3], targets[0][4],
                x_nom[3], x_nom[4]);
            printf("[RENDER] %.4f %.4f %f\n", targets[0][0], targets[0][1], 5.0);
        }

        targets.clear();
        float cur_time = pros::micros() * 1e-6f;
        for (int i = 1; i <= ocp_params.N; i++) {
            Vec target(5);
            float t = cur_time + model_params.dt*i + time_target*1e-3;
            t = modfix(t, spline.get_profile().back().real_time);
            auto point = *std::lower_bound(spline.get_profile().begin(), spline.get_profile().end(), t);
            target << point.pos, 0, point.get_track_speeds(profile_params);
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
        u_nom = pred_actions[0];
        auto target_speed = pred_states[0];

        actuator->volt(left_controller.calculate_voltage(x_nom[3], target_speed[3], u_nom[0]), 
                    right_controller.calculate_voltage(x_nom[4], target_speed[4], u_nom[1]));

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