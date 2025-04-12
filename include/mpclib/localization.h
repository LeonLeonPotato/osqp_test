#pragma once

#include "pros/rtos.h"
#include "pros/rtos.hpp"
#include "utils.h"

namespace mpclib {
class Localization {
    public: 
        virtual float x() = 0;
        virtual float y() = 0;
        virtual float theta() = 0;
        virtual float vl() = 0;
        virtual float vr() = 0;
        Vec get_state(void) {
            return Vec(5);
        }
        std::string to_string() {
            char buffer[4096] = {0};
            sprintf(buffer, "Localization {x = %f, y= %f, θ = %f, vl = %f, vr = %f}", x(), y(), theta(), vl(), vr());
            return std::string(buffer);
        }
};

class SimulatedLocalizer : public Localization {
    private:
        pros::MutexVar<float> x_, y_, theta_, vl_, vr_;
        pros::task_t input_task;

    public:
        SimulatedLocalizer(float x, float y, float theta, float vl = 0, float vr = 0);
        SimulatedLocalizer(void) : mpclib::SimulatedLocalizer(0, 0, 0) {}

        float x() override { return *x_.lock(); }
        float y() override { return *y_.lock(); }
        float theta() override { return *theta_.lock(); }
        float vl() override { return *vl_.lock(); }
        float vr() override { return *vr_.lock(); }
};
}