#include "ocp.h"
#include "hpipm/hpipm_s_ocp_qp_ipm.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace mpclib;

static void create_mask_from_constraints(float* arr, size_t size, const std::vector<Constraint>& constraints) {
    if (arr == nullptr) return;
    std::fill(arr, arr + size, 0.0f);
    for (auto& c : constraints) {
        arr[c.index] = 1.0f;
    }
}

static void create_index_from_constraints(int* arr, size_t size, const std::vector<Constraint>& constraints) {
    if (arr == nullptr) return;
    for (int i = 0; i < size; ++i) {
        arr[i] = constraints[i].index;
    }
}

static void create_bounds_from_constraints(float* lower, float* upper, const std::vector<Constraint>& constraints) {
    if (lower == nullptr || upper == nullptr) return;
    for (int i = 0; i < constraints.size(); ++i) {
        lower[i] = constraints[i].lower_bound;
        upper[i] = constraints[i].upper_bound;
    }
}

template <typename T>
static void print_arr(T* arr, size_t size) {
    std::cout << "[";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i];
        if (i != size - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

static auto autodiff_model_wrapper(const Model& model, const ADVec& x, const ADVec& u) {
    return model.autodiff(x, u);
}

OCPQP::OCPQP(const Model& model, const OCPParams& ocp_params)
    : model(model), ocp_params(ocp_params) 
{
    assert(ocp_params.N > 0);
    assert(ocp_params.Q.rows() == model.state_size());
    assert(ocp_params.Q.cols() == model.state_size());
    assert(ocp_params.R.rows() == model.action_size());
    assert(ocp_params.R.cols() == model.action_size());
    assert(model.state_size() > 0);
    assert(model.action_size() > 0);

    // Allocation of dimension memory
    hpipm_size_t memsize = s_ocp_qp_dim_memsize(ocp_params.N);
    dim_mem = malloc(memsize);
    s_ocp_qp_dim_create(ocp_params.N, &dim, dim_mem);

    setup_dimensions();

    // Allocate internal HPIPM memory for solver itself
    hpipm_size_t qp_size = s_ocp_qp_memsize(&dim);
    qp_mem = malloc(qp_size);
    s_ocp_qp_create(&dim, &qp, qp_mem);

    setup_constraints();
    setup_costs();

    // Additional solver setup (solution, workspace, etc)
    hpipm_size_t qp_sol_size = s_ocp_qp_sol_memsize(&dim);
    qp_sol_mem = malloc(qp_sol_size);
	s_ocp_qp_sol_create(&dim, &qp_sol, qp_sol_mem);

	hpipm_size_t ipm_arg_size = s_ocp_qp_ipm_arg_memsize(&dim);
	ipm_arg_mem = malloc(ipm_arg_size);
	s_ocp_qp_ipm_arg_create(&dim, &ipm_arg, ipm_arg_mem);
	s_ocp_qp_ipm_arg_set_default(hpipm_mode::SPEED, &ipm_arg); // In this case accuracy is not as important as speed
    int warm_start_level = static_cast<int>(ocp_params.warm_start_level);
    s_ocp_qp_ipm_arg_set_warm_start(&warm_start_level, &ipm_arg);
    int it = ocp_params.iterations;
    s_ocp_qp_ipm_arg_set_iter_max(&it, &ipm_arg);

    hpipm_size_t ipm_size = s_ocp_qp_ipm_ws_memsize(&dim, &ipm_arg);
	ipm_mem = malloc(ipm_size);
	s_ocp_qp_ipm_ws_create(&dim, &ipm_arg, &workspace, ipm_mem);
}

void OCPQP::setup_dimensions() {
    // Set the dimensions of the problem
    int state_space[ocp_params.N + 1];
    std::fill_n(state_space, ocp_params.N + 1, model.state_size());

    int action_space[ocp_params.N + 1];
    std::fill_n(action_space, ocp_params.N + 1, model.action_size());
    action_space[ocp_params.N] = 0; // no action at the last timestep as we do not penalize the next state

    int state_bounds[ocp_params.N + 1];
    std::fill_n(state_bounds, ocp_params.N + 1, model.state_constraints().size());
    state_bounds[0] = model.state_size(); // initial state constraint

    int action_bounds[ocp_params.N + 1];
    std::fill_n(action_bounds, ocp_params.N + 1, model.action_constraints().size());
    action_bounds[ocp_params.N] = 0; // no action box constraints at the last timestep

    int general_bounds[ocp_params.N + 1];
    std::fill_n(general_bounds, ocp_params.N + 1, model.general_constraints().size());
    general_bounds[ocp_params.N] = 0; // no general constraints at the last timestep (since no action at last timestep)

    // TODO: implement soft constraining in model
    int soft_bounds[ocp_params.N + 1];
    std::fill_n(soft_bounds, ocp_params.N + 1, 0);

    std::cout << "State space: "; print_arr(state_space, ocp_params.N + 1);
    std::cout << "Action space: "; print_arr(action_space, ocp_params.N + 1);
    std::cout << "State bounds: "; print_arr(state_bounds, ocp_params.N + 1);
    std::cout << "Action bounds: "; print_arr(action_bounds, ocp_params.N + 1);
    std::cout << "General bounds: "; print_arr(general_bounds, ocp_params.N + 1);
    std::cout << "Soft bounds: "; print_arr(soft_bounds, ocp_params.N + 1);

    // Pass into HPIPM
    s_ocp_qp_dim_set_all(state_space, action_space,
        state_bounds, action_bounds, general_bounds,
        soft_bounds, soft_bounds, soft_bounds,
        &dim);

    // Set equality
    s_ocp_qp_dim_set_nbxe(0, model.state_size(), &dim);
}

void OCPQP::setup_constraints() {
    // Create C style constraint masks & indicies from the model
    float state_bound_mask[model.state_size()];
    create_mask_from_constraints(state_bound_mask, model.state_size(), model.state_constraints());

    float action_bound_mask[model.action_size()];
    create_mask_from_constraints(action_bound_mask, model.action_size(), model.action_constraints());

    for(int i = 0; i < ocp_params.N; i++) {
        // State constraints vary from [1, N]
        s_ocp_qp_set_lbx_mask(i+1, state_bound_mask, &qp);
        s_ocp_qp_set_ubx_mask(i+1, state_bound_mask, &qp);

        // Action constraints vary from [0, N-1]
        s_ocp_qp_set_lbu_mask(i, action_bound_mask, &qp);
        s_ocp_qp_set_ubu_mask(i, action_bound_mask, &qp);
    }

    // Avoid UB when creating the index since it could result in zero-sized arrays (I LOVE CPP!!!!)
    if (model.state_constraints().size() > 0) {
        int state_bound_index[model.state_constraints().size()];
        create_index_from_constraints(state_bound_index, model.state_constraints().size(), model.state_constraints());
        std::cout << "State bound index: "; print_arr(state_bound_index, model.state_constraints().size());

        float lower_bound[model.state_constraints().size()];
        float upper_bound[model.state_constraints().size()];
        create_bounds_from_constraints(lower_bound, upper_bound, model.state_constraints());

        // [1, N]
        for(int i = 1; i <= ocp_params.N; i++) {
            s_ocp_qp_set_idxbx(i, state_bound_index, &qp);

            s_ocp_qp_set_lbx(i, lower_bound, &qp);
            s_ocp_qp_set_ubx(i, upper_bound, &qp);
        }
    }

    if (model.action_constraints().size() > 0) {
        int action_bound_index[model.action_constraints().size()];
        create_index_from_constraints(action_bound_index, model.action_constraints().size(), model.action_constraints());
        std::cout << "Action bound index: "; print_arr(action_bound_index, model.action_constraints().size());

        float lower_bound[model.action_constraints().size()];
        float upper_bound[model.action_constraints().size()];
        create_bounds_from_constraints(lower_bound, upper_bound, model.action_constraints());

        // [0, N-1]
        for(int i = 0; i < ocp_params.N; i++) {
            s_ocp_qp_set_idxbu(i, action_bound_index, &qp);

            s_ocp_qp_set_lbu(i, lower_bound, &qp);
            s_ocp_qp_set_ubu(i, upper_bound, &qp);
        }
    }

    if (model.general_constraints().size()) {
        float general_bound_mask[model.general_constraints().size()];
        create_mask_from_constraints(general_bound_mask, model.general_constraints().size(), model.general_constraints());
        std::cout << "General bound mask: "; print_arr(general_bound_mask, model.general_constraints().size());

        float lower_bound[model.general_constraints().size()];
        float upper_bound[model.general_constraints().size()];
        create_bounds_from_constraints(lower_bound, upper_bound, model.general_constraints());

        // General constraints also vary from [0, N-1] (As it depends on action)
        for (int i = 0; i < ocp_params.N; i++) {
            s_ocp_qp_set_lg_mask(i, general_bound_mask, &qp);
            s_ocp_qp_set_ug_mask(i, general_bound_mask, &qp);

            s_ocp_qp_set_C(i, const_cast<float*>(model.general_constraints_state_matrix().data()), &qp);
            s_ocp_qp_set_D(i, const_cast<float*>(model.general_constraints_action_matrix().data()), &qp);

            s_ocp_qp_set_lg(i, lower_bound, &qp);
            s_ocp_qp_set_ug(i, upper_bound, &qp);
        }
    }

    // State mask and index to constrain the initial state (stage=0) to be the current state
    float initial_state_constraint_mask[model.state_size()];
    int initial_state_constraint_index[model.state_size()];
    std::fill_n(initial_state_constraint_mask, model.state_size(), 1.0f);
    std::iota(initial_state_constraint_index, initial_state_constraint_index + model.state_size(), 0);
    s_ocp_qp_set_lbx_mask(0, initial_state_constraint_mask, &qp);
    s_ocp_qp_set_ubx_mask(0, initial_state_constraint_mask, &qp);
    s_ocp_qp_set_idxbx(0, initial_state_constraint_index, &qp);
    s_ocp_qp_set_idxbxe(0, initial_state_constraint_index, &qp);

    std::cout << "x0 mask: "; print_arr(initial_state_constraint_mask, model.state_size());
    std::cout << "x0 index: "; print_arr(initial_state_constraint_index, model.state_size());
}

void OCPQP::setup_costs() {
    Mat zeros = Mat::Zero(model.state_size(), model.state_size());

    // Prevent penalizing the initial state
    s_ocp_qp_set_Q(0, zeros.data(), &qp);
    s_ocp_qp_set_S(0, zeros.data(), &qp);
    s_ocp_qp_set_q(0, zeros.data(), &qp);
    s_ocp_qp_set_r(0, zeros.data(), &qp);

    // Penalize every state from [1, N-1]
    for (int i = 1; i < ocp_params.N; i++) {
        s_ocp_qp_set_Q(i, const_cast<float*>((ocp_params.Q).eval().data()), &qp);
        s_ocp_qp_set_R(i-1, const_cast<float*>(ocp_params.R.data()), &qp);
        s_ocp_qp_set_S(i, zeros.data(), &qp); // No action-state correlation cost
        s_ocp_qp_set_q(i, zeros.data(), &qp); // No linear state cost (Implicitly makes the optimal state equal to the zero vector)
        s_ocp_qp_set_r(i, zeros.data(), &qp); // No linear action cost
    }

    // Final state penalty
    s_ocp_qp_set_Q(ocp_params.N, const_cast<float*>(ocp_params.Qf.data()), &qp);
    s_ocp_qp_set_R(ocp_params.N-1, const_cast<float*>(ocp_params.Rf.data()), &qp);
    s_ocp_qp_set_S(ocp_params.N, zeros.data(), &qp);
    s_ocp_qp_set_q(ocp_params.N, zeros.data(), &qp);
    s_ocp_qp_set_r(ocp_params.N, zeros.data(), &qp);
}

void OCPQP::set_initial_state(const Vec& x) {
    // Its okay (I think) to use const_cast here because the HPIPM API is not const-correct
    s_ocp_qp_set_lbx(0, const_cast<float*>(x.data()), &qp);
    s_ocp_qp_set_ubx(0, const_cast<float*>(x.data()), &qp);
}

void OCPQP::relinearize(const Vec& x, const Vec& u) {
    assert(x.size() == model.state_size());
    assert(u.size() == model.action_size());

    ADVec fout;
    ADVec ad_x = x.cast<autodiff::real>();
    ADVec ad_u = u.cast<autodiff::real>();
    Mat jx = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_x), autodiff::at(model, ad_x, ad_u), fout).cast<float>();
    Mat ju = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_u), autodiff::at(model, ad_x, ad_u), fout).cast<float>();
    Vec c = fout.cast<float>() - (jx * x) - (ju * u);

    // [0, N-1] because the last state (N) does not need to transition to the next (N+1) (It doesnt exist)
    for (int i = 0; i < ocp_params.N; i++) {
        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }
}

void OCPQP::relinearize(Vec x, const std::vector<Vec>& u) {
    ADVec ad_x = x.cast<autodiff::real>();
    Mat jx, ju;
    Vec c;
    for (int i = 0; i < u.size(); i++) {
        ADVec fout;
        ADVec ad_u = u[i].cast<autodiff::real>();
        jx = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_x), autodiff::at(model, ad_x, ad_u), fout).cast<float>();
        ju = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_u), autodiff::at(model, ad_x, ad_u), fout).cast<float>();
        c = fout.cast<float>() - (jx * x) - (ju * u[i]);

        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);

        ad_x = fout.cast<float>().cast<autodiff::real>();
    }

    for (int i = u.size(); i < ocp_params.N; i++) {
        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }
}

void OCPQP::relinearize(const std::vector<Vec>& x, const std::vector<Vec>& u) {
    Mat jx, ju;
    Vec c;

    for (int i = 0; i < std::min(x.size(), u.size()); i++) {
        ADVec fout;
        ADVec ad_x = x[i].cast<autodiff::real>();
        ADVec ad_u = u[i].cast<autodiff::real>();
        jx = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_x), autodiff::at(model, ad_x, ad_u), fout).cast<float>();
        ju = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_u), autodiff::at(model, ad_x, ad_u), fout).cast<float>();
        c = fout.cast<float>() - (jx * x[i]) - (ju * u[i]);

        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }

    for (int i = u.size(); i < ocp_params.N; i++) {
        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }
}

void OCPQP::set_target_state(const Vec& x_desired) {
    Vec q_cost = -ocp_params.Q * x_desired;
    for (int i = 1; i < ocp_params.N; i++) {
        s_ocp_qp_set_q(i, q_cost.data(), &qp);
    }

    Vec qf_cost = -ocp_params.Qf * x_desired;
    s_ocp_qp_set_q(ocp_params.N, qf_cost.data(), &qp);
}

void OCPQP::set_target_state(const std::vector<Vec>& x_desired) {
    if (x_desired.size() != ocp_params.N) {
        throw std::invalid_argument("x_desired size must match the number of timesteps (N)");
    }

    Vec q_cost(model.state_size());
    for (int i = 1; i < ocp_params.N; i++) {
        q_cost << -ocp_params.Q * x_desired[i-1];
        s_ocp_qp_set_q(i, q_cost.data(), &qp);
    }

    Vec qf_cost = -ocp_params.Qf * x_desired[ocp_params.N-1];
    s_ocp_qp_set_q(ocp_params.N, qf_cost.data(), &qp);
}

int OCPQP::solve(bool silent) {
    s_ocp_qp_ipm_solve(&qp, &qp_sol, &ipm_arg, &workspace);

    int status;
    s_ocp_qp_ipm_get_status(&this->workspace, &status);
    if (silent) return status;

    switch (status) {
        case hpipm_status::INCONS_EQ:
            printf("Solving error code %d: INCONS_EQ\n", status);
            break;
        case hpipm_status::NAN_SOL:
            printf("Solving error code %d: NAN_SOL\n", status);
            break;
        default:
            break;
    }
    // MAX_ITER and MIN_STEP are not errors, just warnings

    return status;
}