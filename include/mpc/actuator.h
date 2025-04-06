#pragma once

#include "pros/rtos.h"
#include "pros/rtos.hpp"
#include "utils.h"

namespace mpc {
class Actuator {
    public:
        virtual void volt_left(float voltage) = 0;
        virtual void volt_right(float voltage) = 0;
        virtual void volt(float left, float right) {
            volt_left(left); volt_right(right);
        }
};

class SimulatedActuator : public Actuator {
    private:
        pros::MutexVar<float> last_set_left_volt = 0;
        pros::MutexVar<float> last_set_right_volt = 0;
        pros::task_t output_task;

    public:
        SimulatedActuator(void);

        void volt_left(float voltage);
        void volt_right(float voltage);
};
}