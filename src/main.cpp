#include "main.h"
#include "mpc/localization.h"
#include "mpc/actuator.h"
#include "osqp_test.h"
#include "pros/apix.h"
#include "pros/misc.h"
#include "pros/misc.hpp"


void initialize() {
	// We use serial to communicate, so this needs to be called or else wierd shit happens
	pros::c::serctl(SERCTL_DISABLE_COBS, nullptr);
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}

static inline float minimum_mod_diff(float a, float b, float mod) {
    float diff = fmodf(a - b + mod/2, mod) - mod/2;
    return diff + (diff < -mod/2) * mod;
}

void opcontrol() {
	printf("[RESET]\n");
	pros::Controller master(pros::E_CONTROLLER_MASTER);

	mpc::SimulatedLocalizer localizer;
	mpc::SimulatedActuator actuator;
	float tx = -130;
	float ty = 100;
	while (true) {
		float dx = tx - localizer.x();
		float dy = ty - localizer.y();
		float dist = sqrtf(dx*dx + dy*dy);
		if (dist < 2.54) break;
		float angular_diff = minimum_mod_diff(localizer.theta(), atan2f(dy, dx), M_TWOPI);
		printf("%f\n", localizer.theta());
		float k1 = 0.16;
		float k2 = 5.0;
		
		float forward = k1 * dist * cosf(angular_diff);
		float turn = k2 * angular_diff;
		actuator.volt(forward + turn, forward - turn);
		pros::delay(20);
	}

	actuator.volt(0, 0);
	while (true) { pros::delay(10); }
}