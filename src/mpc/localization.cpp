#include "localization.h"
#include <string>
#include "pros/apix.h"

using namespace mpc;

SimulatedLocalizer::SimulatedLocalizer(float x, float y, float theta, float vl, float vr)
    : x_(x), y_(y), theta_(theta), vl_(vl), vr_(vr) 
{
    input_task = pros::c::task_create([](void* self) {
        SimulatedLocalizer* localizer = static_cast<SimulatedLocalizer*>(self);
        while (true) {
            float tx, ty, ttheta, tvl, tvr;
            std::cin >> tx >> ty >> ttheta >> tvl >> tvr;

            {
                auto x_lock = localizer->x_.lock();
                auto y_lock = localizer->y_.lock();
                auto theta_lock = localizer->theta_.lock();
                auto vl_lock = localizer->vl_.lock();
                auto vr_lock = localizer->vr_.lock();
                
                *x_lock = tx;
                *y_lock = ty;
                *theta_lock = ttheta;
                *vl_lock = tvl;
                *vr_lock = tvr;
            }

            pros::delay(10);
        }
    }, this, TASK_PRIORITY_DEFAULT, TASK_STACK_DEPTH_DEFAULT, "SimulatedLocalizer");
}