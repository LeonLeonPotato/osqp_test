import osqp.utils
import torch
import osqp
import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from typing import Tuple, Union, List

import osqp.tests

class Params:
    def __init__(self, dt, gain, tc, width):
        self.dt = dt
        self.gain = gain
        self.tc = tc
        self.width = width

def f(x, u, params, mod=torch):
    nl = x[3] + (params.gain * u[0] - x[3]) / params.tc * params.dt
    nr = x[4] + (params.gain * u[1] - x[4]) / params.tc * params.dt
    v = (nr + nl) / 2
    omega = (nr - nl) / params.width
    ds = v * params.dt
    dtheta = omega * params.dt
    
    if mod == torch:
        return torch.stack([
            x[0] + ds * torch.cos(x[2] + dtheta/2),
            x[1] + ds * torch.sin(x[2] + dtheta/2),
            x[2] + dtheta,
            nl, nr
        ])
    else:
        return np.array([
            x[0] + ds * np.cos(x[2] + dtheta/2),
            x[1] + ds * np.sin(x[2] + dtheta/2),
            x[2] + dtheta,
            nl, nr
        ])
    
def jac(x, u, params):
    # convert to torch tensor if not already
    x_nom_tensor = x.clone() if isinstance(x, torch.Tensor) else torch.from_numpy(x)
    u_nom_tensor = u.clone() if isinstance(u, torch.Tensor) else torch.from_numpy(u)

    # enable gradient tracking
    x_nom_tensor = x_nom_tensor.detach().requires_grad_(True)
    u_nom_tensor = u_nom_tensor.detach().requires_grad_(True)
    
    # forward pass
    x_next = f(x_nom_tensor, u_nom_tensor, params)
    
    # compute Jacobians
    Jx = []
    for i in range(len(x_next)):
        grad_x = torch.autograd.grad(x_next[i], x_nom_tensor, retain_graph=True, create_graph=False)[0]
        Jx.append(grad_x)
    Jx = torch.stack(Jx, dim=0).detach().numpy()
    
    Ju = []
    for i in range(len(x_next)):
        grad_u = torch.autograd.grad(x_next[i], u_nom_tensor, retain_graph=True, create_graph=False)[0]
        Ju.append(grad_u)
    Ju = torch.stack(Ju, dim=0).detach().numpy()

    c = x_next.detach().numpy() - Jx @ np.array(x) - Ju @ np.array(u)
    return Jx, Ju, c


def alpha_beta(expected:np.ndarray, 
               params:Params, 
               x_nom:np.ndarray, 
               u_nom:List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    N = expected.shape[0]

    Jxs = []; Jus = []; Cs = []

    use_adaptive = True

    if use_adaptive:
        simed_x = x_nom.copy()
        for i in range(N//2):
            Jx_i, Ju_i, c_i = jac(simed_x, u_nom[i], params)
            Jxs.append(Jx_i); Jus.append(Ju_i); Cs.append(c_i)
            simed_x = f(simed_x, u_nom[i], params, np)
            # print(simed_x.numpy())
        while len(Jxs) < N:
            Jxs.append(Jxs[-1]); Jus.append(Jus[-1]); Cs.append(Cs[-1])
    else:
        Jx_i, Ju_i, c_i = jac(x_nom, u_nom[0], params)
        Jxs = [Jx_i for i in range(N)]; Jus = [Ju_i for i in range(N)]; Cs = [c_i for i in range(N)]

    n = Jxs[0].shape[0]; m = Jus[0].shape[1]

    # compute stacked A matrix
    A_block = [Jxs[0]]
    for i in range(1, N):
        A_block.append(Jxs[i] @ A_block[-1])
    A = np.concatenate(A_block, axis=0)

    # compute B matrix
    B1 = [[np.zeros((n, m)) for i in range(N)] for j in range(N)]
    B2C = [[np.zeros((n, n)) for i in range(N)] for j in range(N)]
    for j in range(N):
        track = Jus[j]
        track2 = np.eye(n)
        B1[j][j] = track
        B2C[j][j] = track2
        for i in range(j+1, N):
            track = Jxs[i] @ track
            B1[i][j] = track
            track2 = Jxs[i] @ track2
            B2C[i][j] = track2
    B1 = np.block(B1)
    B2C = np.block(B2C)

    B2C = B2C @ np.concatenate(Cs, axis=0)

    # print(A.shape, Jx.shape, x_nom.reshape(-1, 1).shape, C.reshape(-1, 1).shape, expected.flatten().reshape(-1, 1).shape)

    alpha = A @ x_nom.reshape(-1, 1) + B2C.reshape(-1, 1) - expected.flatten().reshape(-1, 1)
    beta = B1
    return alpha, beta

def mpc(expected, params, x_nom, u_nom, Q, R, Qf, max_change=3, max_value=12):
    N = expected.shape[0]
    n = x_nom.shape[0]; m = u_nom[0].shape[0]
    alpha, beta = alpha_beta(expected, params, x_nom, u_nom[1:])

    # test_u = np.array([
    #     12.0, 0,
    #     12.0, 0,
    #     12.0, 0
    # ]).reshape(-1, 1)
    # test_x = np.array([
    #     0.0, 0.0, np.pi/2, 0.0, 0.0
    # ]).reshape(-1, 1)
    # X = (alpha + beta @ test_u).reshape(-1, 5)
    # print(alpha.shape, beta.shape, test_u.shape, X.shape)
    # np.set_printoptions(suppress=True,precision=3)
    # print(X)
    # exit()


    # Q = [np.exp(Q) / np.exp(N-1) for i in range(N-1)]
    Q = [Q for i in range(N-1)]
    # Q = [Q for i in range(N-1)]
    Q = sparse.block_diag(Q + [Qf])
    R = sparse.kron(sparse.eye(N), R)

    prob = osqp.OSQP()

    P = sparse.csc_matrix(2 * (beta.T @ Q @ beta + R))
    q = 2 * beta.T @ Q @ alpha

    eye = sparse.eye(N * m, format='csc')
    diff = -eye.copy()
    diff.setdiag(1, 2)
    A = sparse.vstack([
        eye, diff
    ], format='csc')

    bl = np.full((N * m,), -max_value)
    bl[0] = max(-max_value, u_nom[0][0] - max_change)
    bl[1] = max(-max_value, u_nom[0][1] - max_change)
    bl = np.concatenate([bl, np.full((N * m,), -max_change)])

    bu = np.full((N * m,), max_value)
    bu[0] = min(max_value, u_nom[0][0] + max_change)
    bu[1] = min(max_value, u_nom[0][1] + max_change)
    bu = np.concatenate([bu, np.full((N * m,), max_change)])
    
    prob.setup(P, q, 
                A=A,
                l=bl,
                u=bu,
               verbose=False)
    res = prob.solve()
    return res.x

if __name__ == "__main__":
    N = 30
    x_nom = np.array([0, 0, -np.pi/2, 0.0, 0.0])
    u_nom = [np.random.randn(2) for i in range(N)]
    Q = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
    R = np.eye(2) * 0.0001
    # Qf = np.diag([0.0, 0.0, 0.0, 0.0, 0.0])
    Qf = Q
    expected = np.zeros((N, x_nom.shape[0]), dtype=np.float32)
    expected[:, :] = [50, 50, np.pi/2, 0, 0]

    sim_steps = 300
    mpcdt = 0.04
    realdt = 0.02

    path_x = CubicSpline(
        np.linspace(0, 5, 4),
        np.array([122.3, 20.8, -96.5, -105.0]),
        bc_type='natural'
    )
    path_y = CubicSpline(
        np.linspace(0, 5, 4),
        np.array([39.4, -39.0, 86.9, 9.4]),
        bc_type='natural'
    )

    params = Params(
        dt=0.01,
        gain=6.03627741577 * 4.125 * 0.75,
        tc=0.162619135755,
        width=30.54
    )

    time = 0
    xy = []
    theta = []
    targets = []
    for i in range(sim_steps):
        h = np.linspace(time, time + mpcdt*N, N)
        expected[:, 0] = path_x(h)
        expected[:, 1] = path_y(h)
        targets.append((expected[0, 0], expected[0, 1]))

        params.dt = mpcdt
        u = mpc(expected, params, x_nom, u_nom, Q, R, Qf)
        params.dt = realdt

        u_nom = [np.array([u[i], u[i+1]]) for i in range(0, len(u), 2)]
        output_u = u_nom[0].copy()
        # u_nom[0] = np.clip(u_nom[0], -12, 12); u_nom[1] = np.clip(u_nom[1], -12, 12)
        old_theta = x_nom[2]
        x_nom = f(x_nom, output_u, params, np)
        new_theta = x_nom[2]
        x_nom[2] = old_theta + (new_theta - old_theta) * np.random.normal(1.0, 0.5)

        print(f"ts {i+1}: {output_u[0]:.2f}, {output_u[1]:.2f} | {x_nom[0]:.2f}, {x_nom[1]:.2f}, {x_nom[2]:.2f}")
        xy.append(tuple(x_nom[:2].round(2)))
        theta.append(x_nom[2].round(2))

        time += realdt
    print("P = " + str(xy))
    # print("T = " + str(theta))
    # print("O = " + str(targets))
    