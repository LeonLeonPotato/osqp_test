#include "base_model.h"

namespace mpclib {
class DifferentialDriveModel final : public DriveModel {
public:
    static constexpr int GENERAL_CONSTRAINTS_SIZE = 2; // left wheel speed, right wheel speed
    static constexpr int STATE_BOX_CONSTRAINTS_SIZE = 0; // No state box constraints
    static constexpr int ACTION_BOX_CONSTRAINTS_SIZE = 0; // No action box constraints

    struct Params : BaseParams {
        float width; // Drivetrain width; the distance between the left and right drivetrains, in cm. Must be positive.
        float max_speed; // Maximum (absolute value) speed of one side of the drivetrain on the robot, in cm / s. Must be positive.
        float acceleration_constant; // How fast robot accelerates, in s ^ -1. Larger -> faster acceleration. Must be positive.
    };

    DifferentialDriveModel(const Params& params);

    constexpr int get_number_state_box_constraints() const override { return STATE_BOX_CONSTRAINTS_SIZE; }
    constexpr int get_number_action_box_constraints() const override { return ACTION_BOX_CONSTRAINTS_SIZE; }
    constexpr int get_number_general_constraints() const override { return GENERAL_CONSTRAINTS_SIZE; }

    ADVec autodiff(const ADVec& x, const ADVec& u) const override;
    Vec infer(const Vec& x, const Vec& u) const override;

    const Params& get_params() const { return params; }
    Params& get_params_nonconst() { return params; }
    void set_params(const Params& params) { this->params = params; }

    float* get_general_lower_bound() override;
    float* get_general_upper_bound() override;

    int* get_general_lower_bound_mask() override { return general_lower_bound_mask; }
    int* get_general_upper_bound_mask() override { return general_upper_bound_mask; }

    float* get_general_constraints_state_matrix() override;
    float* get_general_constraints_action_matrix() override;

private:
    Params params;

    float general_lower_bound[GENERAL_CONSTRAINTS_SIZE] = {0, 0};
    float general_upper_bound[GENERAL_CONSTRAINTS_SIZE] = {0, 0};
    int general_lower_bound_mask[GENERAL_CONSTRAINTS_SIZE] = {1, 1};
    int general_upper_bound_mask[GENERAL_CONSTRAINTS_SIZE] = {1, 1};
    
    Mat general_constraints_state_matrix;
    Mat general_constraints_action_matrix;
};
}