#include "base_model.h"

namespace mpclib {
class UnicycleModel : public Model {
public: 
    struct Params : BaseParams {
        float max_linear_speed; // Maximum linear travel speed of the robot, in cm / s
        float max_angular_speed; // Maximum angular (turning) speed of the robot, in radian / s
        float max_linear_acceleration; // Maximum (absolute value) acceleration of the robot, in cm / s^2
        float max_angular_acceleration; // Maximum (absolute value) angular acceleration of the robot, in radian / s^2
    };

    UnicycleModel(const Params& params) : params(params) {};

    ADVec autodiff(const ADVec& x, const ADVec& u) const override;
    Vec infer(const Vec& x, const Vec& u) const override;

    const Params& get_params() const { return params; }
    Params& get_params_nonconst() { return params; }
    void set_params(const Params& params) { this->params = params; }

    virtual float* get_state_lower_bound() override;
    virtual float* get_state_upper_bound() override;
    virtual int* get_state_lower_bound_mask() override;
    virtual int* get_state_upper_bound_mask() override;

    virtual float* get_action_lower_bound() override;
    virtual float* get_action_upper_bound() override;
    virtual int* get_action_lower_bound_mask() override;
    virtual int* get_action_upper_bound_mask() override;

    virtual float* get_general_lower_bound() override;
    virtual float* get_general_upper_bound() override;
    virtual int* get_general_lower_bound_mask() override;
    virtual int* get_general_upper_bound_mask() override;

private:
    Params params;
};
}