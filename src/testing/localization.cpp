#include "testing/localization.h"
#include <string>
#include "pros/apix.h"

using namespace mpclib;

static inline std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    tokens.push_back(str.substr(start, end));
    return tokens;
}

SimulatedLocalizer::SimulatedLocalizer(float x, float y, float theta, float vl, float vr)
    : x_(x), y_(y), theta_(theta), vl_(vl), vr_(vr) 
{
    input_task = pros::c::task_create([](void* self) {
        SimulatedLocalizer* localizer = static_cast<SimulatedLocalizer*>(self);
        while (true) {
            float tx, ty, ttheta, tvl, tvr;
            std::string line;
            std::getline(std::cin, line);
            auto tokens = split(line, ' ');
            tx = std::stof(tokens[0]);
            ty = std::stof(tokens[1]);
            ttheta = std::stof(tokens[2]);
            tvl = std::stof(tokens[3]);
            tvr = std::stof(tokens[4]);

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

SimulatedLocalizer::~SimulatedLocalizer(void) {
    pros::c::task_delete(input_task);
}