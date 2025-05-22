#include "pros/misc.hpp"
#include "pros/motor_group.hpp"
#include "testing/test.h"

using namespace testing;

#define SCALE (127.0f / 12.0f)

constexpr float turn_sensitivity = 0.5f;

void testing::driver() {
	pros::Controller master(pros::E_CONTROLLER_MASTER);
	pros::MotorGroup left_motors({-1, 2, -3});
	pros::MotorGroup right_motors({8, 10, -21});

	while (true) {
		int left_x = master.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_X);
		int left_y = master.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y);
		int right_x = master.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X);
		int right_y = master.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_Y);
		float left = std::clamp(left_y + (int) (right_x * turn_sensitivity), -127, 127) / SCALE;
		float right = std::clamp(left_y - (int) (right_x * turn_sensitivity), -127, 127) / SCALE;
		// actuator->volt(left, right);
		left_motors.move_voltage(left * 12000);
		right_motors.move_voltage(right * 12000);
		pros::delay(10);
	}

	actuator->volt(0, 0);
	while (true) { pros::delay(10); }
}