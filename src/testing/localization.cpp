#include "testing/localization.h"
#include <string>
#include "pros/apix.h"
#include "pros/imu.hpp"

using namespace mpclib;

#define rad(x) ((x) * M_PI / 180.0f)

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

Odometry::Odometry(float x, float y, float theta, 
    int forward_encoder_port, int sideways_encoder_port,
    float forward_encoder_offset, float sideways_encoder_offset,
    DT left_motors, DT right_motors, pros::IMU imu,
    float tracking_wheel_radius, float velocity_multiplier)
        : x_(x), y_(y), theta_(theta), 
        forward_encoder_port(forward_encoder_port), sideways_encoder_port(sideways_encoder_port),
        forward_encoder_offset(forward_encoder_offset), sideways_encoder_offset(sideways_encoder_offset),
        left_motors(left_motors), right_motors(right_motors), imu(imu),
        tracking_wheel_radius(tracking_wheel_radius), velocity_multiplier(velocity_multiplier) 
{
    task = pros::c::task_create([](void* self) {
        Odometry* odometry = static_cast<Odometry*>(self);
        long long ltime = pros::micros();
        float ltheta = odometry->imu.get_rotation();
        float ll = rad(pros::c::rotation_get_position(odometry->forward_encoder_port) / 100.0f);
        float lh = rad(pros::c::rotation_get_position(odometry->sideways_encoder_port) / 100.0f);

        pros::delay(10);

        while (true) {
            auto theta_lock = odometry->theta_.lock();
            auto x_lock = odometry->x_.lock();
            auto y_lock = odometry->y_.lock();

            float dt = (pros::micros() - ltime) / 1e6f;
            ltime = pros::micros();

            float ctheta = odometry->imu.get_rotation();
            float dtheta = ctheta - ltheta;
            ltheta = ctheta;

            *theta_lock = ctheta;

            float cl = rad(pros::c::rotation_get_position(odometry->forward_encoder_port) / 100.0f) * odometry->tracking_wheel_radius;
            float ch = rad(pros::c::rotation_get_position(odometry->sideways_encoder_port) / 100.0f) * odometry->tracking_wheel_radius;
            float travel_forwards = (cl - ll);
            float travel_horizontal = (ch - lh);
            ll = cl;
            lh = ch;

            if (dtheta != 0) {
                float constant = 2 * sin(dtheta / 2);
                travel_forwards = constant * (travel_forwards / dtheta + odometry->forward_encoder_offset);
                travel_horizontal = constant * (travel_horizontal / dtheta + odometry->sideways_encoder_offset);
            }

            float av_theta = ctheta - dtheta / 2;

            float dx = travel_forwards * cosf(av_theta) - travel_horizontal * sinf(av_theta);
            float dy = travel_forwards * sinf(av_theta) + travel_horizontal * cosf(av_theta);
            *x_lock += dx;
            *y_lock += dy;

            pros::delay(10);
        }
    }, this, TASK_PRIORITY_DEFAULT, TASK_STACK_DEPTH_DEFAULT, "Odometry");
}