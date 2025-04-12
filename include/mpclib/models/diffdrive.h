#include "base_model.h"

namespace mpclib {
class DifferentialDriveModel final : public DriveModel {
public:
    static constexpr int GENERAL_CONSTRAINTS_SIZE = 2; // left wheel speed, right wheel speed
    static constexpr int STATE_CONSTRAINTS_SIZE = 0; // No state box constraints
    static constexpr int ACTION_CONSTRAINTS_SIZE = 0; // No action box constraints

    struct Params : BaseParams {
        float width; // Drivetrain width; the distance between the left and right drivetrains, in cm. Must be positive.
        float max_speed; // Maximum (absolute value) speed of one side of the drivetrain on the robot, in cm / s. Must be positive.
        float acceleration_constant; // How fast robot accelerates, in s ^ -1. Larger -> faster acceleration. Must be positive.
    };

    DifferentialDriveModel(const Params& params);

    ADVec autodiff(const ADVec& x, const ADVec& u) const override;
    Vec infer(const Vec& x, const Vec& u) const override;

    const Params& params() const { return params_; }
    void set_params(const Params& params) { params_ = params; resync_from_params(); }

private:
    Params params_;

    void resync_from_params();
};
}