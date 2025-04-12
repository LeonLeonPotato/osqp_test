#include "testing/testing.h"
#include <cstdio>
#include "mpclib/models.h"
#include "mpclib/ocp.h"
#include "pros/rtos.hpp"

using namespace mpclib;

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
    model_params.dt = 0.2f;
    model_params.width = 34.5f;
    model_params.max_speed = 150.0f;
    model_params.acceleration_constant = 2.4f;
    DifferentialDriveModel model(model_params);

    DifferentialDriveModel::Params simulation_params;
    simulation_params.dt = 0.1f;
    simulation_params.width = 34.5f;
    simulation_params.max_speed = 150.0f;
    simulation_params.acceleration_constant = 2.4f;
    DifferentialDriveModel simulator(simulation_params);

    OCPParams ocp_params;
    ocp_params.N = 30;
    ocp_params.Q = (Eigen::Vector<float, 5> {1, 1, 0, 0, 0}).asDiagonal();
    ocp_params.R = Mat::Zero(2, 2);
    ocp_params.warm_start_level = OCPParams::WarmStartLevel::STATE;

    Vec x0(5); x0 << 0, 0, 0.1, 1, 0;
    Vec u0(2); u0 << 1, 0;
    Vec x_target(5) ; x_target << -100, 100, 0, 0, 0;

    OCPQP ocpqp(model, ocp_params);
    ocpqp.set_initial_state(x0);
    ocpqp.relinearize(x0, u0);
    ocpqp.set_target_state(x_target);

    // Simulate the model with the solution
    for (int i = 0; i < 100; i++) {
        auto t1 = pros::micros();
        int status = ocpqp.solve(false);
        auto t2 = pros::micros();
        printf("Step %d Solve time: %lld us         ", i, t2 - t1);

        s_ocp_qp_sol_get_u(0, &ocpqp.qp_sol, u0.data());
        x0 = simulator.infer(x0, u0);
        
        ocpqp.set_initial_state(x0);
        ocpqp.relinearize(x0, u0);

        Vec u_buffer(2); Vec x_buffer(5);
        for (int j = ocpqp.ocp_params.N; j > 1; j--) {
            s_ocp_qp_sol_get_x(j, &ocpqp.qp_sol, x_buffer.data());
            s_ocp_qp_sol_get_u(j-1, &ocpqp.qp_sol, u_buffer.data());

            s_ocp_qp_sol_set_x(j-1, x_buffer.data(), &ocpqp.qp);
            s_ocp_qp_set_ubx(j, x_buffer.data(), &ocpqp.qp);
        }

        printf("Action: [%.2f, %.2f] State: [%.3f, %.3f, %.3f, %.3f, %.3f]\n", u0[0], u0[1], x0[0], x0[1], x0[2], x0[3], x0[4]);
        printf("%d\n", ocpqp.ipm_arg.warm_start);
    }
}