#include "testing/test.h" // IWYU pragma: keep
#include "mpclib/models.h"
#include "mpclib/ocp.h"

using namespace testing;

void testing::ocp_qp() {
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