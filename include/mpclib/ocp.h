/**
 * @file ocp.h
 * @author Leon
 * @brief Wrapper class of the Optimal Control Problem Quadratic Programming (OCP-QP) solver using HPIPM.
 *
 * @details
 * This file defines the structures and methods for solving optimal control problems
 * using the HPIPM library. It includes the definition of OCP parameters, the OCPQP
 * class for managing the problem setup, and methods for relinearization, setting
 * targets, and solving the problem.
 */

#pragma once

#include "models/base_model.h"
#include "utils.h"
#include "hpipm/hpipm_s_ocp_qp.h"
#include "hpipm/hpipm_s_ocp_qp_dim.h"
#include "hpipm/hpipm_s_ocp_qp_sol.h"
#include "hpipm/hpipm_s_ocp_qp_ipm.h"

namespace mpclib {

/**
 * @brief Structure holding configuration parameters for the OCP solver.
 * @note
 * All matricies are expected to be in the correct shape. This means that 
 * all state cost (Q) matricies are of size (state_size, state_size),
 * all action cost (R) matricies are of size (action_size, action_size), 
 * and all state-action correlation (S) matricies are of size (state_size, action_size).
 */
struct OCPParams {
    int N; ///< Number of time steps in the OCP problem. This means there are N actions & N states to predict excluding the initial state.

    Mat Q1; ///< State cost for stage 1 (\f$ x_1 \f$).
    Mat Q;  ///< State cost for intermediate stages 2 … N‑1.
    Mat Qf; ///< State cost for stage N (\f$ x_N \f$)

    Mat R0; ///< Action cost for stage 0 (\f$ u_0 \f$).
    Mat R;  ///< Action cost for intermediate stages 1 … N‑2.
    Mat Rf; ///< Action cost for stage (\f$ u_{N-1} \f$).

    /**
     * @brief Enum defining warm start levels.
     */
    enum class WarmStartLevel {
        NONE = 0, ///< No warm start. Both state and input are "guessed" to be zero.
        STATE = 1, ///< Only the state is warm started. The input is "guessed" to be zero.
        STATE_AND_INPUT = 2, ///< Both state and input are warm started, i.e. reused from the previous solve.
        FULL = 3 ///< Full warm start, including state, input, other IPM variables.
    } warm_start_level = WarmStartLevel::STATE; ///< Default – reuse state only.

    int iterations = 5; ///< Maximum number of iterations per solve call
};

/**
 * @brief Class wrapping the OCP-QP solver.
 * @details
 * Handles initialization and solution of OCP problems using HPIPM. Provides functions to set targets,
 * initial conditions, constraints, and cost functions.
 */
struct OCPQP {
    const OCPParams& ocp_params; ///< Reference to OCP configuration.
    const Model& model; ///< Reference to the model defining system dynamics and constraints.

    s_ocp_qp_dim dim; ///< HPIPM dimension object.
    void* dim_mem; ///< Memory for HPIPM dimension object.

    s_ocp_qp qp; ///< HPIPM QP object.
    void* qp_mem; ///< Memory for HPIPM QP object.

    s_ocp_qp_sol qp_sol; ///< HPIPM solution object.
    void* qp_sol_mem; ///< Memory for HPIPM solution object.

    s_ocp_qp_ipm_ws workspace; ///< HPIPM workspace object.
    void* ipm_mem; ///< Memory for HPIPM workspace.

    s_ocp_qp_ipm_arg ipm_arg; ///< HPIPM solver argument object.
    void* ipm_arg_mem; ///< Memory for HPIPM solver argument object.

    /**
     * @brief Construct a partially initialized QP wrapper.
     *
     * @details
     * All HPIPM structures are allocated and the *static* parts of the problem
     * (dimensions, box constraints, state or action independent cost weights) are set up.
     * No linearization or initial state setting is performed here – call one 
     * of the `relinearize()` overloads and set the initial state before solving.
     *
     * @param model Dynamics model
     * @param ocp_params  Horizon length, cost weights and solver options.
     */
    OCPQP(const Model& model, const OCPParams& ocp_params);

    /**
     * @brief Constrain the current state
     * 
     * @details
     * In HPIPM, the initial state is set as both the lower and upper bounds 
     * of the state variable. This constrains all future states, which are 
     * states in stages from \f$ [1, N] \f$ to evolve from the initial state.
     * 
     * @param x Initial state vector.
     */
    void set_initial_state(const Vec& x);

    /**
     * @brief Relinearize the model dynamics using a single state and input.
     * 
     * @details
     * This function will compute the jacobian of the model dynamics
     * and set the linearized dynamics in the QP problem. This is used to
     * approximate nonlinear models with a linear one - specifically, a
     * first order taylor expansion, where 
     * \f$ f(x, u) \approx J_{x}x + J_{u}u + [f(x_{nom}, u_{nom}) - J_{x}x_{nom} - J_{u}u_{nom}] \f$ 
     * 
     * @param x State vector to linearize around.
     * @param u Action vector to linearize around.
     * 
     * @see @ref mpclib::Model::autodiff
     */
    void relinearize(const Vec& x, const Vec& u);

    /**
     * @brief Relinearize the model dynamics by evolving a state from a sequence of actions
     * 
     * @details 
     * Similar to @ref relinearize, but this function will evolve the state
     * using the model dynamics @ref mpclib::Model::infer and the sequence of actions. This essentially "predicts"
     * the future states of the model, and moves the operating points to the predicted states each state.
     * This prevents errors  from linearization in highly non-linear models, when \f$ x \f$ is far from \f$ x_{nom} \f$,
     * or when \f$ u \f$ is far from \f$ u_{nom} \f$.
     * 
     * @param x the initial state to evolve from.
     * @param u list of actions to evolve the state with.
     *
     * @see @ref mpclib::Model::autodiff
     * @note If the size of `u` is less than `N`, the last state & action will be used to fill the rest of the dynamics matricies.
     */
    void relinearize(Vec x, const std::vector<Vec>& u);

    /**
     * @brief Relinearize the model dynamics from a sequence of states and actions
     * 
     * @details
     * Similar to @ref relinearize, but this function will set the dynamics matricies at each stage
     * to the ones computed from the state and action pairs. Note that this differs from 
     * @ref relinearize(Vec x, const std::vector<Vec>& u) in that it does not evolve any states.
     * 
     * @param x States to linearize around at each stage.
     * @param u Actions to linearize around at each stage.
     *
     * @see @ref mpclib::Model::autodiff
     * @note If the size of `u` or `x` is less than `N`, the last state & action will be used to fill the rest of the dynamics matricies.
     * Additionally, if the size of `x` differs from `u`, the smaller of the two will be used.
     */
    void relinearize(const std::vector<Vec>& x, const std::vector<Vec>& u);

    /**
     * @brief Set a single target state across all N stages
     * 
     * @param x_desired Desired state vector to achieve
     */
    void set_target_state(const Vec& x_desired);

    /**
     * @brief Set possibly unique target states in each stage
     * 
     * @details
     * At each stage \f$ i \f$ in the range \f$ [1, N] \f$, the target state is set to the \f$ i-1 \f$ element in the list.
     *
     * @param x_desired List of desired states to achieve at each stage
     */
    void set_target_state(const std::vector<Vec>& x_desired);
    void set_target_input(const Vec& u_desired);
    void set_target_input(const std::vector<Vec>& u_desired);

    int solve(bool silent = true);

private:
    void setup_dimensions();
    void setup_constraints();
    void setup_costs();
};
}