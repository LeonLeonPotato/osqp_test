# VEX MPC

⚠️ This project is very much in the experimental phase and I cannot guarantee that it will work ⚠️

![Source: https://www.mathworks.com/help/mpc/gs/what-is-mpc.html](https://www.mathworks.com/help/mpc/gs/mpc-intro-structure.png)
<p align="center">Illustration of Model Predictive Control</p>

Vex MPC aims to bring [model predictive control (MPC)](https://www.mathworks.com/help/mpc/gs/what-is-mpc.html) as a method of motion control to VEX V5.
The rational for this is that at its core, MPC relies on a physical model of the system to be controlled to optimize over.
This results in a continuously improvable movement algorithm just by improving the model itself.
For example, the model can range from a simple unicycle model with `[x, y, theta]` state and  `[v, omega]` input variables to a full-on physics simulation.

This is especially applicable for omni-wheeled robots, as they often do not have many movement algorithms centered around them due to their difficulty
in accounting for sideways slippage.

Another reason to use MPC is its ability to handle non-holonomic constraints.
In VEX, most drivetrains are non-holonomic, meaning that meaning that the robot cannot move in arbitrary directions instantaneously.
For example, a differential drive robot (like a standard VEX chassis with two drivetrains) cannot strafe sideways—it
can only move forward/backward and rotate. This restriction is non-integrable, meaning it cannot be expressed purely as constraints on position,
but rather on the velocities of the system.

## Documentation

The project is fully documented with `doxygen`. To generate HTML documentation, run `doxygen Doxyfile` in the project's base path.
Then, go to `docs/html/index.html` to get to the docs.

Online-hosted documentation will come later.

## Formulation

This library uses [HPIPM](https://github.com/giaf/hpipm) as its core solver and [BLASFEO](https://github.com/giaf/blasfeo) as its backend BLAS library.

Because many systems, including this one, are non-linear in nature, they cannot be directly optimized using nonlinear methods as this is extremely
expensive. Instead, a better way is to linearize around potentially multiple operating points by the use of a first-order taylor approximation.

With this approximation, it turns out that the problem can be formulated into a [quadratic programming (QP)](https://en.wikipedia.org/wiki/Quadratic_programming) problem.
In this case, the QP has a specific structure called an **O**ptimal **C**ontrol **P**roblem (OCP). Below is the formal problem as stated in the [HPIPM paper](https://publications.syscop.de/Frison2020a.pdf):

![HPIPM OCP QP formulation](https://imgur.com/w4yq9us.png)
<p align="center">HPIPM OCP QP formulation</p>

HPIPM exploits this structure to efficiently solve the QP. Its speed is increased with the use of BLASFEO, which is optimized for embedded performance
by the use of placing small matricies on the cache instead of RAM.

## Library

Currently the `main` branch uses condensed OCP QP formulation which means that the state variables are directly expressed as `alpha + beta * u` in the objective function.

This 'decision' was made because I didn't know about OSQP being sparse-only and how in general OCP-QP scale horribly (in terms of horizon length) with condensed formulation.

To amend these mistakes I have decided to rewrite in HPIPM and sparse formulation.