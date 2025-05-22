#pragma once

#include "pros/imu.hpp"
#include "pros/motors.h"
#include "pros/motors.hpp"
#include "pros/rotation.h"
#include "pros/rtos.h"
#include "pros/rtos.hpp"
#include "mpclib/utils.h"

namespace mpclib {
class Localization {
    public: 
        virtual float x() = 0;
        virtual float y() = 0;
        virtual float theta() = 0;
        virtual float vl() = 0;
        virtual float vr() = 0;
        Vec get_state(void) {
            Vec state(5);
            state << x(), y(), theta(), vl(), vr();
            return state;
        }
        std::string to_string() {
            char buffer[256] = {0};
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
        ~SimulatedLocalizer(void);

        float x() override { return *x_.lock(); }
        float y() override { return *y_.lock(); }
        float theta() override { return *theta_.lock(); }
        float vl() override { return *vl_.lock(); }
        float vr() override { return *vr_.lock(); }
};

class Odometry : public Localization {
    private:
        typedef std::vector<pros::Motor*> DT;

        int forward_encoder_port, sideways_encoder_port;
        float forward_encoder_offset, sideways_encoder_offset;
        float tracking_wheel_radius;
        DT left_motors, right_motors;
        pros::IMU imu;
        float velocity_multiplier; // Angular * velo mult = linear velo
        pros::MutexVar<float> x_, y_, theta_;
        pros::task_t task;

    public:
        Odometry(float x, float y, float theta, 
            int forward_encoder_port, int sideways_encoder_port,
            float forward_encoder_offset, float sideways_encoder_offset,
            DT left_motors, DT right_motors, pros::IMU imu,
            float tracking_wheel_radius, float velocity_multiplier = 1.0f);

        Odometry(int forward_encoder_port, int sideways_encoder_port,
            float forward_encoder_offset, float sideways_encoder_offset,
            DT left_motors, DT right_motors, pros::IMU imu,
            float tracking_wheel_radius, float velocity_multiplier = 1.0f) 
                : mpclib::Odometry(0, 0, 0,
                forward_encoder_port, sideways_encoder_port,
                forward_encoder_offset, sideways_encoder_offset,
                left_motors, right_motors, imu,
                tracking_wheel_radius, velocity_multiplier) {}
        
        ~Odometry(void);

        float x() override { return *x_.lock(); }
        float y() override { return *y_.lock(); }
        float theta() override { return *theta_.lock(); }
        float vl() override { 
            std::vector<float> left_velocities;
            for (auto motor : left_motors) {
                left_velocities.push_back(motor->get_actual_velocity());
            }
            return average(left_velocities) * velocity_multiplier;
        }
        float vr() override {
            std::vector<float> right_velocities;
            for (auto motor : right_motors) {
                right_velocities.push_back(motor->get_actual_velocity());
            }
            return average(right_velocities) * velocity_multiplier;
        }
};
}