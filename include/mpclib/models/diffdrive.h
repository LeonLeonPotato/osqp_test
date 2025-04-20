/**
 * @file diffdrive.h
 * @author Leon
 * @brief Differential drive model
 * @date 2025-04-20
 */

#include "base_model.h"

namespace mpclib {

/**
 * @brief A model representing a differential drive robot.
 *
 * @details
 * This class implements the dynamics of a differential drive system, such as a typical two-wheeled ground robot.
 * It extends the DriveModel base class and defines both the dynamics and constraints for the system.
 * 
 * The model supports general constraints on the individual wheel speeds (left and right), which are derived
 * from the maximum allowed drivetrain speed. It does not impose box constraints on the state or action vectors directly.
 *
 * The model parameters define physical properties such as drivetrain width, max wheel speed, and acceleration behavior.
 *
 * @see mpclib::DriveModel
 */
class DifferentialDriveModel final : public DriveModel {
public:
    static constexpr int GENERAL_CONSTRAINTS_SIZE = 2; ///< left wheel speed, right wheel speed
    static constexpr int STATE_CONSTRAINTS_SIZE = 0; ///< No state box constraints
    static constexpr int ACTION_CONSTRAINTS_SIZE = 0; ///< No action box constraints

    /**
    * @brief Parameters specific to the differential drive robot model.
    *
    * @details
    * These parameters define the physical and behavioral properties of the robot. They are used to configure
    * the model's dynamics and constraints. All parameters must be positive.
    *
    * @param width Width of the drivetrain, i.e., the distance between the left and right wheels (in cm).
    * @param max_speed Maximum allowed wheel speed for either side of the drivetrain (in cm/s).
    * @param acceleration_constant A tuning parameter controlling how quickly the robot accelerates. 
    *        Higher values mean faster response (in seconds).
    *
    * @see mpclib::Model::BaseParams
    */
    struct Params : BaseParams {
        float width; ///< Drivetrain width; the distance between the left and right drivetrains, in cm. Must be positive.
        float max_speed; ///< Maximum (absolute value) speed of one side of the drivetrain on the robot, in cm / s. Must be positive.
        float acceleration_constant; ///< How fast robot accelerates, in seconds. Larger -> faster acceleration. Must be positive.
    };

    /**
    * @brief Constructor for the DifferentialDriveModel.
    *
    * @param params Struct containing all the model parameters required in the dynamics and constraints.
    */
    DifferentialDriveModel(const Params& params);

    /// @copydoc mpclib::Model::autodiff
    ADVec autodiff(const ADVec& x, const ADVec& u) const override;
    /// @copydoc mpclib::Model::infer
    Vec infer(const Vec& x, const Vec& u, float dt_override = -1) const override;

    /**
     * @brief Get the parameters of the model.
     * 
     * @return `const Params&` - Constant reference to the model parameters.
     * @see mpclib::DifferentialDriveModel::Params 
     */
    const Params& params() const { return params_; }

    /**
     * @brief Set the parameters of the model.
     * 
     * @param params The new parameters to set for the model.
     * @see mpclib::DifferentialDriveModel::Params 
     */
    void set_params(const Params& params) { params_ = params; resync_from_params(); }

private:
    Params params_; ///< The parameters of the model

    /**
     * @brief Syncs all internally stored variables to @ref params_.
     * 
     * @details
     * This function is called whenever the parameters are changed. It updates all internal variables
     * that depend on the parameters, such as @ref general_state_matrix and @ref general_action_matrix.
     * 
     * It is called automatically in the constructor and when the parameters are set using @ref set_params.
     */
    void resync_from_params();
};
}