/**
 * @file ocp.h
 * @brief Wrapper class of the Optimal Control Problem Quadratic Programming (OCP-QP) solver using HPIPM.
 *
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
struct OCPParams {
    int N;
    Mat Q1, Q, Qf;
    Mat R0, R, Rf;

    enum class WarmStartLevel {
        NONE = 0,
        STATE = 1,
        STATE_AND_INPUT = 2,
        FULL = 3
    } warm_start_level = WarmStartLevel::STATE;
    int iterations = 5;
};

struct OCPQP {
    const OCPParams& ocp_params;
    const Model& model;

    s_ocp_qp_dim dim;
    void* dim_mem;

    s_ocp_qp qp;
    void* qp_mem;

    s_ocp_qp_sol qp_sol;
    void* qp_sol_mem;

    s_ocp_qp_ipm_ws workspace;
    void* ipm_mem;

    s_ocp_qp_ipm_arg ipm_arg;
    void* ipm_arg_mem;

    OCPQP(const Model& model, const OCPParams& ocp_params);

    void set_initial_state(const Vec& x_nom);

    void relinearize(const Vec& x, const Vec& u);
    void relinearize(Vec x, const std::vector<Vec>& u);
    void relinearize(const std::vector<Vec>& x, const std::vector<Vec>& u);

    void set_target_state(const Vec& x_desired);
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