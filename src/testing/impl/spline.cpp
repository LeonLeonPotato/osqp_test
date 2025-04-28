#include "testing/test.h" // IWYU pragma: keep
#include "testing/n_degree_spline.h"

using namespace testing;
using namespace pathing;

void testing::spline() {
    pathing::CubicSpline spline {
        {{0, 0}, {100, 100}, {110, 0}, {-50, -50}, {-100, -100}, {-110, 0}, {0, 0}}
    };
    // spline.solve_coeffs(
    //     {pathing::Condition{1, M_PI/2, 250}}, 
    //     {pathing::Condition{1, M_PI/2, 250}}
    // );
    spline.solve_coeffs(
        {pathing::CubicSpline::natural_conditions},
        {pathing::CubicSpline::natural_conditions}
    );

    printf("Spline: %s\n", spline.debug_out().c_str());

    // for (int i = 10; i <= 300; i += 10) {
    //     float t = spline.path_parametrize(i);
    //     float s = spline.length(t);
    //     printf("inquiry: %f, t: %f, calc s: %f\n", (float) i, t, s);
    // }

    float gain = 13.6; float kf = 0.7;
    auto profile_start = pros::micros();
    spline.profile_path({
        .max_speed = (gain - kf) * 12.0f,
        .max_accel = 110.0f,
        .min_speed = 1.0f,
        .track_width = 33.87f,
        .dt = 0.01f
    });
    auto profile_end = pros::micros();
    printf("Profile time: %f ms\n", (profile_end - profile_start) / 1000.0f);

    std::vector<Vec> buf;
    for (auto& p : spline.get_profile()) {
        buf.push_back((Vec(2) << p.distance, p.speed).finished());
    }

    print_vectors(buf, 3, true, "p");

    pros::delay(10);

    std::cout << "\n\n\n";

    std::vector<float> buf2;
    for (auto& p : spline.get_profile()) {
        buf2.push_back(p.path_param);
    }

    print_vector(buf2, 3, true, "u");
}