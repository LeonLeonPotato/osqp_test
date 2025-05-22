#include "testing/actuator.h"

using namespace mpclib;

SimulatedActuator::SimulatedActuator(void) {
    output_task = pros::c::task_create([](void* self) {
        SimulatedActuator* actuator = static_cast<SimulatedActuator*>(self);
        while (true) {
            float left, right;
            {
                auto left_lock = actuator->last_set_left_volt.lock();
                auto right_lock = actuator->last_set_right_volt.lock();
                left = *left_lock;
                right = *right_lock;
            }
            std::cout << "[INPUTS] " << left << " " << right << std::endl;
            pros::delay(10);
        }
    }, this, TASK_PRIORITY_DEFAULT, TASK_STACK_DEPTH_DEFAULT, "SimulatedActuator");
}

SimulatedActuator::~SimulatedActuator(void) {
    pros::c::task_delete(output_task);
}

void SimulatedActuator::volt_left(float left) {
    left = std::clamp(left, -12.0f, 12.0f);
    *this->last_set_left_volt.lock() = left;
}

void SimulatedActuator::volt_right(float right) {
    right = std::clamp(right, -12.0f, 12.0f);
    *this->last_set_right_volt.lock() = right;
}