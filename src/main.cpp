#include "main.h"
#include "mpclib/localization.h"
#include "mpclib/actuator.h"
#include "pros/apix.h"
#include "pros/misc.h"
#include "pros/misc.hpp"

#define SCALE (127.0f / 12.0f)

void initialize() {
	// We use serial to communicate, so this needs to be called or else wierd shit happens
	pros::c::serctl(SERCTL_DISABLE_COBS, nullptr);
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}

void opcontrol() {

}

void driver() {
	printf("[RESET]\n");
	pros::Controller master(pros::E_CONTROLLER_MASTER);

	mpclib::SimulatedLocalizer localizer;
	mpclib::SimulatedActuator actuator;

	while (true) {
		int left_x = master.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_X);
		int left_y = master.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y);
		int right_x = master.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X);
		int right_y = master.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_Y);
		float left = std::clamp(left_y + right_x, -127, 127) / SCALE;
		float right = std::clamp(left_y - right_x, -127, 127) / SCALE;
		actuator.volt(left, right);
		pros::delay(20);
	}

	actuator.volt(0, 0);
	while (true) { pros::delay(10); }
}