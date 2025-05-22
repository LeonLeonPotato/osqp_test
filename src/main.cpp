#include "main.h"
#include "pros/apix.h"
#include "testing/localization.h"
#include "testing/actuator.h"
#include "testing/test.h"

void initialize() {
	// We use serial to communicate, so this needs to be called or else wierd shit happens
	pros::c::serctl(SERCTL_DISABLE_COBS, nullptr);
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}

SimulatedActuator* testing::actuator;
SimulatedLocalizer* testing::localizer;

void opcontrol() {
	printf("[RESET]\n");
	// pros::delay(200);

	// testing::actuator = new mpclib::SimulatedActuator();
	// testing::localizer = new mpclib::SimulatedLocalizer();

	// testing::simulation();
	testing::driver();
	// testing::spline();
}