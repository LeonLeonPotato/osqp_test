#include "mpclib/models.h"
#include "testing/test.h" // IWYU pragma: keep

using namespace testing;

void testing::model() {
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