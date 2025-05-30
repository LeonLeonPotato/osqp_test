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



#ifndef HPIPM_S_COND_AUX_H_
#define HPIPM_S_COND_AUX_H_



#include "blasfeo/blasfeo_target.h"
#include "blasfeo/blasfeo_common.h"



#ifdef __cplusplus
extern "C" {
#endif



//
void s_cond_BAbt(struct s_ocp_qp *ocp_qp, struct blasfeo_smat *BAbt2, struct blasfeo_svec *b2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_BAt(struct s_ocp_qp *ocp_qp, struct blasfeo_smat *BAbt2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_b(struct s_ocp_qp *ocp_qp, struct blasfeo_svec *b2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_RSQrq(struct s_ocp_qp *ocp_qp, struct blasfeo_smat *RSQrq2, struct blasfeo_svec *rq2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_RSQ(struct s_ocp_qp *ocp_qp, struct blasfeo_smat *RSQrq2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_rq(struct s_ocp_qp *ocp_qp, struct blasfeo_svec *rq2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_DCtd(struct s_ocp_qp *ocp_qp, int *idxb2, struct blasfeo_smat *DCt2, struct blasfeo_svec *d2, struct blasfeo_svec *d_mask2, int *idxs_rev2, struct blasfeo_svec *Z2, struct blasfeo_svec *z2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_DCt(struct s_ocp_qp *ocp_qp, int *idxb2, struct blasfeo_smat *DCt2, int *idxs_rev2, struct blasfeo_svec *Z2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_d(struct s_ocp_qp *ocp_qp, struct blasfeo_svec *d2, struct blasfeo_svec *d_mask2, struct blasfeo_svec *z2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_cond_sol(struct s_ocp_qp *ocp_qp, struct s_ocp_qp_sol *ocp_qp_sol, struct s_dense_qp_sol *dense_qp_sol, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_expand_sol(struct s_ocp_qp *ocp_qp, struct s_dense_qp_sol *dense_qp_sol, struct s_ocp_qp_sol *ocp_qp_sol, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);

//
void s_update_cond_BAbt(int *idxc, struct s_ocp_qp *ocp_qp, struct blasfeo_smat *BAbt2, struct blasfeo_svec *b2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_update_cond_RSQrq_N2nx3(int *idxc, struct s_ocp_qp *ocp_qp, struct blasfeo_smat *RSQrq2, struct blasfeo_svec *rq2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);
//
void s_update_cond_DCtd(int *idxc, struct s_ocp_qp *ocp_qp, int *idxb2, struct blasfeo_smat *DCt2, struct blasfeo_svec *d2, int *idxs2, struct blasfeo_svec *Z2, struct blasfeo_svec *z2, struct s_cond_qp_arg *cond_arg, struct s_cond_qp_ws *cond_ws);



#ifdef __cplusplus
} /* extern "C" */
#endif



#endif // HPIPM_S_COND_AUX_H_
