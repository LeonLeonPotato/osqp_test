/**************************************************************************************************
*                                                                                                 *
* This file is part of HPIPM.                                                                     *
*                                                                                                 *
* HPIPM -- High-Performance Interior Point Method.                                                *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/


#ifndef HPIPM_S_TREE_OCP_QP_KKT_H_
#define HPIPM_S_TREE_OCP_QP_KKT_H_



#include "hpipm_common.h"
#include "hpipm_s_tree_ocp_qp.h"
#include "hpipm_s_tree_ocp_qp_sol.h"
#include "hpipm_d_tree_ocp_qp_ipm.h"



#ifdef __cplusplus
extern "C" {
#endif

//
void s_tree_ocp_qp_fact_solve_kkt_unconstr(struct s_tree_ocp_qp *qp, struct s_tree_ocp_qp_sol *qp_sol, struct s_tree_ocp_qp_ipm_arg *arg, struct s_tree_ocp_qp_ipm_ws *ws);
//
void s_tree_ocp_qp_fact_solve_kkt_step(struct s_tree_ocp_qp *qp, struct s_tree_ocp_qp_sol *qp_sol, struct s_tree_ocp_qp_ipm_arg *arg, struct s_tree_ocp_qp_ipm_ws *ws);
//
void s_tree_ocp_qp_fact_lq_solve_kkt_step(struct s_tree_ocp_qp *qp, struct s_tree_ocp_qp_sol *qp_sol, struct s_tree_ocp_qp_ipm_arg *arg, struct s_tree_ocp_qp_ipm_ws *ws);
//
void s_tree_ocp_qp_solve_kkt_step(struct s_tree_ocp_qp *qp, struct s_tree_ocp_qp_sol *qp_sol, struct s_tree_ocp_qp_ipm_arg *arg, struct s_tree_ocp_qp_ipm_ws *ws);


#ifdef __cplusplus
} /* extern "C" */
#endif



#endif // HPIPM_S_TREE_OCP_QP_KKT_H_
