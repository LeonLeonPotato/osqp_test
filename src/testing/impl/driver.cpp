#include "pros/misc.hpp"
#include "testing/test.h"

using namespace testing;

#define SCALE (127.0f / 12.0f)

void testing::driver() {
	pros::Controller master(pros::E_CONTROLLER_MASTER);

	while (true) {
		int left_x = master.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_X);
		int left_y = master.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y);
		int right_x = master.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X);
		int right_y = master.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_Y);
		float left = std::clamp(left_y + right_x, -127, 127) / SCALE;
		float right = std::clamp(left_y - right_x, -127, 127) / SCALE;
		actuator->volt(left, right);
		pros::delay(20);
	}

	actuator->volt(0, 0);
	while (true) { pros::delay(10); }
}