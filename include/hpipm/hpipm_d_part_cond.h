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



#ifndef HPIPM_D_PART_COND_H_
#define HPIPM_D_PART_COND_H_



#include "blasfeo/blasfeo_target.h"
#include "blasfeo/blasfeo_common.h"

#include "hpipm_common.h"
#include "hpipm_d_cond.h"


#ifdef __cplusplus
extern "C" {
#endif



struct d_part_cond_qp_arg
	{
	struct d_cond_qp_arg *cond_arg;
	int N2;
	hpipm_size_t memsize;
	};



struct d_part_cond_qp_ws
	{
	struct d_cond_qp_ws *cond_workspace;
	hpipm_size_t memsize;
	};



//
hpipm_size_t d_part_cond_qp_arg_memsize(int N2);
//
void d_part_cond_qp_arg_create(int N2, struct d_part_cond_qp_arg *cond_arg, void *mem);
//
void d_part_cond_qp_arg_set_default(struct d_part_cond_qp_arg *cond_arg);
// set riccati-like algorithm: 0 classical, 1 squre-root
void d_part_cond_qp_arg_set_ric_alg(int ric_alg, struct d_part_cond_qp_arg *cond_arg);
//
void d_part_cond_qp_arg_set_comp_prim_sol(int value, struct d_part_cond_qp_arg *cond_arg);
//
void d_part_cond_qp_arg_set_comp_dual_sol_eq(int value, struct d_part_cond_qp_arg *cond_arg);
//
void d_part_cond_qp_arg_set_comp_dual_sol_ineq(int value, struct d_part_cond_qp_arg *cond_arg);

//
void d_part_cond_qp_compute_block_size(int N, int N2, int *block_size);
//
void d_part_cond_qp_compute_dim(struct d_ocp_qp_dim *ocp_dim, int *block_size, struct d_ocp_qp_dim *part_dense_dim);
//
hpipm_size_t d_part_cond_qp_ws_memsize(struct d_ocp_qp_dim *ocp_dim, int *block_size, struct d_ocp_qp_dim *part_dense_dim, struct d_part_cond_qp_arg *cond_arg);
//
void d_part_cond_qp_ws_create(struct d_ocp_qp_dim *ocp_dim, int *block_size, struct d_ocp_qp_dim *part_dense_dim, struct d_part_cond_qp_arg *cond_arg, struct d_part_cond_qp_ws *cond_ws, void *mem);
//
void d_part_cond_qp_cond(struct d_ocp_qp *ocp_qp, struct d_ocp_qp *part_dense_qp, struct d_part_cond_qp_arg *cond_arg, struct d_part_cond_qp_ws *cond_ws);
//
void d_part_cond_qp_cond_lhs(struct d_ocp_qp *ocp_qp, struct d_ocp_qp *part_dense_qp, struct d_part_cond_qp_arg *cond_arg, struct d_part_cond_qp_ws *cond_ws);
//
void d_part_cond_qp_cond_rhs(struct d_ocp_qp *ocp_qp, struct d_ocp_qp *part_dense_qp, struct d_part_cond_qp_arg *cond_arg, struct d_part_cond_qp_ws *cond_ws);
//
void d_part_cond_qp_cond_sol(struct d_ocp_qp *ocp_qp, struct d_ocp_qp *part_dense_qp, struct d_ocp_qp_sol *ocp_qp_sol, struct d_ocp_qp_sol *part_dense_qp_sol, struct d_part_cond_qp_arg *cond_arg, struct d_part_cond_qp_ws *cond_ws);
//
void d_part_cond_qp_expand_sol(struct d_ocp_qp *ocp_qp, struct d_ocp_qp *part_dense_qp, struct d_ocp_qp_sol *part_dense_qp_sol, struct d_ocp_qp_sol *ocp_qp_sol, struct d_part_cond_qp_arg *cond_arg, struct d_part_cond_qp_ws *cond_ws);

//
void d_part_cond_qp_update(int *idxc, struct d_ocp_qp *ocp_qp, struct d_ocp_qp *part_dense_qp, struct d_part_cond_qp_arg *cond_arg, struct d_part_cond_qp_ws *cond_ws);


#ifdef __cplusplus
} /* extern "C" */
#endif



#endif // HPIPM_D_PART_COND_H_
