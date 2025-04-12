#include "mpclib/testing.h"
#include <cstdio>
#include "mpclib/models.h"

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
    float* g_sm = model.get_general_constraints_state_matrix();
    float* g_am = model.get_general_constraints_action_matrix();
    Mat* g_sm_wtf = (Mat*) (reinterpret_cast<char*>(&model) + 164);
    Mat* g_am_wtf = (Mat*) (reinterpret_cast<char*>(&model) + 176);

    std::cout << "General Constraints State Matrix:\n";
    std::cout << *g_sm_wtf << std::endl;
    std::cout << "General Constraints Action Matrix:\n";
    std::cout << *g_am_wtf << std::endl;
}
