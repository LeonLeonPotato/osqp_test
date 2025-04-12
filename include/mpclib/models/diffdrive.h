#include "base_model.h"

namespace mpclib {
class DifferentialDriveModel : public DriveModel {
public:
    struct Params : BaseParams {
        float width; // Drivetrain width; the distance between the left and right drivetrains, in cm
        float max_acceleration; // Maximum (absolute value) acceleration of one side of the drivetrain on the robot, in cm / s^2
        float max_speed; // Maximum (absolute value) speed of one side of the drivetrain on the robot, in cm / s
        float time_constant; // How slow robot accelerates, in seconds. Lower -> faster acceleration.
    };

    DifferentialDriveModel(const Params& params) : params(params) {};

    ADVec autodiff(const ADVec& x, const ADVec& u) const override;
    Vec infer(const Vec& x, const Vec& u) const override;

    const Params& get_params() const { return params; }
    Params& get_params_nonconst() { return params; }
    void set_params(const Params& params) { this->params = params; }

    virtual float* get_state_lower_bound() override;
    virtual float* get_state_upper_bound() override;

    virtual float* get_action_lower_bound() override;
    virtual float* get_action_upper_bound() override;

    virtual float* get_general_lower_bound() override;
    virtual float* get_general_upper_bound() override;

private:
    Params params;
};
}