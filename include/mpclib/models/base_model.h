#pragma once

#include "mpclib/utils.h"

namespace mpclib {
struct Constraint {
    float lower_bound;
    float upper_bound;
    int index;
};

class Model {
public: 
    struct BaseParams {
        float dt; // Time step, in seconds
    };

    virtual constexpr int state_size() const = 0;
    virtual constexpr int action_size() const = 0;

    virtual ADVec autodiff(const ADVec& x, const ADVec& u) const = 0;
    virtual Vec infer(const Vec& x, const Vec& u) const = 0;

    virtual const std::vector<Constraint>& state_constraints() const { return state_constraints_; }
    virtual const std::vector<Constraint>& action_constraints() const { return action_constraints_; }
    virtual const std::vector<Constraint>& general_constraints() const { return general_constraints_; }

    virtual const Mat& general_constraints_state_matrix() const { return general_state_matrix; }
    virtual const Mat& general_constraints_action_matrix() const { return general_action_matrix; }

protected:
    std::vector<Constraint> state_constraints_;
    std::vector<Constraint> action_constraints_;
    std::vector<Constraint> general_constraints_;

    Mat general_state_matrix; // constraints matrix
    Mat general_action_matrix; // constraints matrix
};

class DriveModel : public Model {
public:
    static constexpr int STATE_SIZE = 5; // x, y, theta, [velocities]
    static constexpr int ACTION_SIZE = 2; // [accelerations]

    constexpr int state_size() const override { return STATE_SIZE; }
    constexpr int action_size() const override { return ACTION_SIZE; }
};
}