# VEX MPC
***"Im still making ts"*** 

This project is under construction \*tm*

## Formulation
Currently the `main` branch uses condensed OCP QP formulation which means that the state variables are directly expressed as `alpha + beta * u` in the objective function.

This 'decision' was made because I didn't know about OSQP being sparse-only and how in general OCP-QP scale horribly (in terms of horizon length) with condensed formulation.

To amend these mistakes I have decided to rewrite in HPIPM and sparse formulation.

## qpOASES test branch
Ignore this as it was a failed test for qpOASES. It was way too slow.