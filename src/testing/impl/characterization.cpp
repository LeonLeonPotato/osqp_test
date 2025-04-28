#include "testing/test.h"

using namespace testing;

void testing::find_center() {
    actuator->volt(-12, 12);
    while (true) {
        auto x = localizer->get_state();
        printf("x: [%.3f, %.3f, %.3f]\n", x[0], x[1], x[2]);
        pros::c::task_delay(10);
    }
}

void testing::motor_constants() {
    int stage = 8;
    actuator->volt(stage / 10.0f * 12, stage / 10.0f * 12);
    std::vector<Vec> positions;

    auto start = pros::micros();
    auto last_t = start;
    float last_x = localizer->get_state()[0];
    pros::delay(20);
    for (int i = 0; i < 50; i++) {
        auto x = localizer->x();
        auto t = (pros::micros() - start) * 1e-6f;
        auto dt = (pros::micros() - last_t) * 1e-6f;
        auto dx = x - last_x;
        last_x = x;
        last_t = pros::micros();
        positions.push_back((Vec(2) << t, (dx/dt)).finished());
        pros::delay(20);
    }

    print_vectors(positions);

    while (true) pros::delay(50);
}