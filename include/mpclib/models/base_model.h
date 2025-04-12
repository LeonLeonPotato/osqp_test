#pragma once

#include "mpclib/utils.h"

namespace mpclib {
class Model {
public: 
    struct BaseParams {
        float dt; // Time step, in seconds
    };

    virtual constexpr int get_state_size() const = 0;
    virtual constexpr int get_action_size() const = 0;
    virtual constexpr int get_number_state_box_constraints() const = 0;
    virtual constexpr int get_number_action_box_constraints() const = 0;
    virtual constexpr int get_number_general_constraints() const = 0;

    virtual ADVec autodiff(const ADVec& x, const ADVec& u) const = 0;
    virtual Vec infer(const Vec& x, const Vec& u) const = 0;

    virtual float* get_state_lower_bound() { return nullptr; }
    virtual float* get_state_upper_bound() { return nullptr; }
    virtual int* get_state_lower_bound_mask() { return nullptr; }
    virtual int* get_state_upper_bound_mask() { return nullptr; }

    virtual float* get_action_lower_bound() { return nullptr; }
    virtual float* get_action_upper_bound() { return nullptr; }
    virtual int* get_action_lower_bound_mask() { return nullptr; }
    virtual int* get_action_upper_bound_mask() { return nullptr; }

    virtual float* get_general_lower_bound() { return nullptr; }
    virtual float* get_general_upper_bound() { return nullptr; }
    virtual int* get_general_lower_bound_mask() { return nullptr; }
    virtual int* get_general_upper_bound_mask() { return nullptr; }

    virtual float* get_general_constraints_state_matrix() { return nullptr; }
    virtual float* get_general_constraints_action_matrix() { return nullptr; }
};

class DriveModel : public Model {
public:
    static constexpr int STATE_SIZE = 5; // x, y, theta, [velocities]
    static constexpr int ACTION_SIZE = 2; // [accelerations]

    constexpr int get_state_size() const override { return STATE_SIZE; }
    constexpr int get_action_size() const override { return ACTION_SIZE; }

    virtual float* get_state_lower_bound() override { return state_lower_bound; }
    virtual float* get_state_upper_bound() override { return state_upper_bound; }
    virtual int* get_state_lower_bound_mask() override { return state_lower_bound_mask; }
    virtual int* get_state_upper_bound_mask() override { return state_upper_bound_mask; }

    virtual float* get_action_lower_bound() override { return action_lower_bound; }
    virtual float* get_action_upper_bound() override { return action_upper_bound; }
    virtual int* get_action_lower_bound_mask() override { return action_lower_bound_mask; }
    virtual int* get_action_upper_bound_mask() override { return action_upper_bound_mask; }

protected:
    // NONE of this can be const because HPIPM for some reason expects non-const arrays?
    float state_lower_bound[5] = {0, 0, 0, 0, 0};
    float state_upper_bound[5] = {0, 0, 0, 0, 0};
    int state_lower_bound_mask[5] = {0, 0, 0, 1, 1};
    int state_upper_bound_mask[5] = {0, 0, 0, 1, 1};

    float action_lower_bound[2] = {0, 0};
    float action_upper_bound[2] = {0, 0};
    int action_lower_bound_mask[2] = {1, 1};
    int action_upper_bound_mask[2] = {1, 1};
};
}