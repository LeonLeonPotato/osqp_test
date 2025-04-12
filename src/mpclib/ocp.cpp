#include "ocp.h"
#include <algorithm>
#include <memory>
#include <vector>

using namespace mpclib;

static int count_nonzeros(int* arr, int size) {
    if (arr == nullptr) return 0;
    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (arr[i] != 0) count++;
    }
    return count;
}

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


OCPQP::OCPQP(const Model& model, const OCPParams& ocp_params)
    : model(model), ocp_params(ocp_params) 
{
    // Part 1 dimensions setup
    {
        // Allocation of dimension memory
        hpipm_size_t memsize = s_ocp_qp_dim_memsize(ocp_params.N);
        dim_mem = malloc(memsize);
        s_ocp_qp_dim_create(ocp_params.N, &dim, dim_mem);

        // Set the dimensions of the problem
        int state_space[ocp_params.N + 1];
        std::fill_n(state_space, dim.N + 1, model.state_size());

        int action_space[ocp_params.N + 1];
        std::fill_n(action_space, dim.N + 1, model.action_size());
        action_space[dim.N] = 0; // no action at the last timestep as we do not penalize the next state

        int state_bounds[ocp_params.N + 1];
        std::fill_n(state_bounds, dim.N + 1, model.state_constraints().size());
        state_bounds[0] = model.state_size(); // initial state constraint

        int action_bounds[ocp_params.N + 1];
        std::fill_n(action_bounds, dim.N + 1, model.action_constraints().size());
        action_bounds[dim.N] = 0; // no action box constraints at the last timestep

        int general_bounds[ocp_params.N + 1];
        std::fill_n(general_bounds, dim.N + 1, model.general_constraints().size());

        // TODO: implement soft constraining in model
        int soft_bounds[ocp_params.N + 1];
        std::fill_n(soft_bounds, dim.N + 1, 0);

        // Pass into HPIPM
        s_ocp_qp_dim_set_all(state_space, action_space,
            state_bounds, action_bounds, general_bounds,
            soft_bounds, soft_bounds, soft_bounds,
            &dim);
    }

    // Part 2 problem formulation
    {
        // Allocate internal HPIPM memory for solution
        hpipm_size_t qp_size = s_ocp_qp_memsize(&dim);
        qp_mem = malloc(qp_size);
        s_ocp_qp_create(&dim, &qp, qp_mem);

        // Create C style constraint masks & indicies from the model
        float state_bound_mask[model.state_size()];
        int state_bound_index[model.state_constraints().size()];
        create_mask_from_constraints(state_bound_mask, model.state_size(), model.state_constraints());
        create_index_from_constraints(state_bound_index, model.state_constraints().size(), model.state_constraints());

        float action_bound_mask[model.action_size()];
        int action_bound_index[model.action_constraints().size()];
        create_mask_from_constraints(action_bound_mask, model.action_size(), model.action_constraints());
        create_index_from_constraints(action_bound_index, model.action_constraints().size(), model.action_constraints());

        float general_bound_mask[model.general_constraints().size()];
        create_mask_from_constraints(general_bound_mask, model.general_constraints().size(), model.general_constraints());

        for(int i = 0; i < ocp_params.N; i++) {
            // State constraints vary from [1, N]
            s_ocp_qp_set_lbx_mask(i+1, state_bound_mask, &qp);
            s_ocp_qp_set_ubx_mask(i+1, state_bound_mask, &qp);
            s_ocp_qp_set_idxbx(i+1, state_bound_index, &qp);

            // Action constraints vary from [0, N-1]
            s_ocp_qp_set_lbu_mask(i, action_bound_mask, &qp);
            s_ocp_qp_set_ubu_mask(i, action_bound_mask, &qp);
            s_ocp_qp_set_idxbu(i, action_bound_index, &qp);

            // General constraints vary from [0, N-1] (As it depends on action)
            s_ocp_qp_set_lg_mask(i, general_bound_mask, &qp);
            s_ocp_qp_set_ug_mask(i, general_bound_mask, &qp);
        }

        // State mask and index to constrain the initial state (stage=0) to be the current state
        float initial_state_constraint_mask[model.state_size()];
        int initial_state_constraint_index[model.state_size()];
        std::fill_n(initial_state_constraint_mask, model.state_size(), 1.0f);
        std::iota(initial_state_constraint_index, initial_state_constraint_index + model.state_size(), 0);
        s_ocp_qp_set_lbx_mask(0, initial_state_constraint_mask, &qp);
        s_ocp_qp_set_ubx_mask(0, initial_state_constraint_mask, &qp);
        s_ocp_qp_set_idxbx(0, initial_state_constraint_index, &qp);
    }

    // Part 3 solver setup
    hpipm_size_t qp_sol_size = s_ocp_qp_sol_memsize(&dim);
    qp_sol_mem = malloc(qp_sol_size);
	s_ocp_qp_sol_create(&dim, &qp_sol, qp_sol_mem);

	hpipm_size_t ipm_arg_size = s_ocp_qp_ipm_arg_memsize(&dim);
	ipm_arg_mem = malloc(ipm_arg_size);
	s_ocp_qp_ipm_arg_create(&dim, &ipm_arg, ipm_arg_mem);
	s_ocp_qp_ipm_arg_set_default(hpipm_mode::SPEED, &ipm_arg); // In this case accuracy is not as important as speed

    hpipm_size_t ipm_size = s_ocp_qp_ipm_ws_memsize(&dim, &ipm_arg);
	ipm_mem = malloc(ipm_size);
	s_ocp_qp_ipm_ws_create(&dim, &ipm_arg, &workspace, ipm_mem);
}