import osqp.utils
import torch
import osqp
import numpy as np
from scipy import sparse
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
    x_nom_tensor = x_nom.clone() if isinstance(x, torch.Tensor) else torch.from_numpy(x)
    u_nom_tensor = u_nom.clone() if isinstance(u, torch.Tensor) else torch.from_numpy(u)

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

    c = x_next.detach().numpy() - Jx @ x_nom - Ju @ u_nom
    return Jx, Ju, c


def alpha_beta(expected:np.ndarray, 
               params:Params, 
               x_nom:Union[np.ndarray, torch.Tensor], 
               u_nom:List[Union[np.ndarray, torch.Tensor]]) -> Tuple[np.ndarray, np.ndarray]:
    N = expected.shape[0]

    Jx = []; Ju = []; C = []

    simed_x = x_nom if isinstance(x_nom, torch.Tensor) else torch.from_numpy(x_nom)
    for i in range(N):
        Jx_i, Ju_i, c_i = jac(simed_x, u_nom[i], params)
        Jx.append(Jx_i); Ju.append(Ju_i); C.append(c_i)
        simed_x = f(simed_x, u_nom[i], params, torch).detach()

    n = Jx[0].shape[0]
    m = Ju[0].shape[0]

    # compute stacked A matrix
    A_block = [np.eye(n)]
    for i in range(N - 1):
        A_block.append(A_block[-1] @ Jx[i])
    A = np.concatenate(A_block, axis=0)

    # compute B matrix
    B1 = [[np.zeros_like(Ju) for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(i+1):
            B1[i][j] = A_block[i - j] @ Ju
    B1 = np.block(B1)

    C = [c]
    for i in range(1, N):
        C.append(C[-1] + (A_block[i] @ c))
    C = np.concatenate(C, axis=0)

    # print(A.shape, Jx.shape, x_nom.reshape(-1, 1).shape, C.reshape(-1, 1).shape, expected.flatten().reshape(-1, 1).shape)

    alpha = A @ Jx @ x_nom.reshape(-1, 1) + C.reshape(-1, 1) - expected.flatten().reshape(-1, 1)
    beta = B1
    return alpha, beta

def mpc(expected, params, x_nom, u_nom, Q, R, Qf, max_change=3, max_value=12):
    N = expected.shape[0]
    alpha, beta = alpha_beta(expected, params, x_nom, u_nom)

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
    Q = [Q * np.exp(0.01 * i) for i in range(N-1)]
    # Q = [Q for i in range(N-1)]
    Q = sparse.block_diag(Q + [Qf])
    R = sparse.kron(sparse.eye(N), R)

    prob = osqp.OSQP()

    P = sparse.csc_matrix(2 * (beta.T @ Q @ beta + R))
    q = 2 * beta.T @ Q @ alpha

    eye = sparse.eye(N * u_nom.shape[0], format='csc')
    diff = -eye.copy()
    diff.setdiag(1, 2)
    A = sparse.vstack([
        eye, diff
    ], format='csc')

    bl = np.full((N * u_nom.shape[0],), -max_value)
    bl[0] = max(-max_value, u_nom[0] - max_change)
    bl[1] = max(-max_value, u_nom[1] - max_change)
    bl = np.concatenate([bl, np.full((N * u_nom.shape[0],), -max_change)])

    bu = np.full((N * u_nom.shape[0],), max_value)
    bu[0] = min(max_value, u_nom[0] + max_change)
    bu[1] = min(max_value, u_nom[1] + max_change)
    bu = np.concatenate([bu, np.full((N * u_nom.shape[0],), max_change)])
    
    prob.setup(P, q, 
                A=A,
                l=bl,
                u=bu,
               verbose=False)
    res = prob.solve()
    return res.x

if __name__ == "__main__":
    x_nom = np.array([-50, 50, -np.pi/2, 0.0, 0.0])
    u_nom = np.array([0.0, 0.0])
    Q = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
    R = np.eye(2) * 0.0001
    # Qf = np.diag([0.0, 0.0, 0.0, 0.0, 0.0])
    Qf = Q
    N = 20
    expected = np.zeros((N, x_nom.shape[0]), dtype=np.float32)
    expected[:, :] = [50, 50, np.pi/2, 0, 0]

    params = Params(
        dt=0.01,
        gain=6.03627741577 * 4.125 * 0.75,
        tc=0.162619135755,
        width=30.54
    )

    mpcdt = 0.06
    realdt = 0.04

    time = 0
    xy = []
    theta = []
    targets = []
    for i in range(150):
        h = np.linspace(time, time + mpcdt*N, N)
        expected[:, 0] = 50 * np.sqrt(2) * np.cos(h) / (np.sin(h)**2 + 1)
        expected[:, 1] = 50 * np.sqrt(2) * np.cos(h) * np.sin(h) / (np.sin(h)**2 + 1)
        targets.append((expected[0, 0], expected[0, 1]))

        params.dt = mpcdt
        u = mpc(expected, params, x_nom, u_nom, Q, R, Qf)
        params.dt = realdt

        u_nom = np.array([u[0], u[1]])
        # u_nom[0] = np.clip(u_nom[0], -12, 12); u_nom[1] = np.clip(u_nom[1], -12, 12)
        x_nom = f(x_nom, u_nom, params, np)
        # x_nom[2] += np.random.normal(0, 0.2)

        print(f"ts {i+1}: {u[0]:.2f}, {u[1]:.2f} | {x_nom[0]:.2f}, {x_nom[1]:.2f}, {x_nom[2]:.2f}")
        xy.append(tuple(x_nom[:2].round(2)))
        theta.append(x_nom[2].round(2))

        time += realdt
    print("P = " + str(xy))
    print("T = " + str(theta))
    print("O = " + str(targets))
    