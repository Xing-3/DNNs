import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import math
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional
import matplotlib.pyplot as plt


# ---------------------------------- Parameters ------------------------------------#
alpha = 0.5
N = 15
T = 1
delta = T / N
epsil = 1e-5
pi = np.pi

_a = (1.00000000000000000000, 0.57721566490153286061, -0.65587807152025388108,
      -0.04200263503409523553, 0.16653861138229148950, -0.04219773455554433675,
      -0.00962197152787697356, 0.00721894324666309954, -0.00116516759185906511,
      -0.00021524167411495097, 0.00012805028238811619, -0.00002013485478078824,
      -0.00000125049348214267, 0.00000113302723198170, -0.00000020563384169776,
      0.00000000611609510448, 0.00000000500200764447, -0.00000000118127457049,
      0.00000000010434267117, 0.00000000000778226344, -0.00000000000369680562,
      0.00000000000051003703, -0.00000000000002058326, -0.00000000000000534812,
      0.00000000000000122678, -0.00000000000000011813, 0.00000000000000000119,
      0.00000000000000000141, -0.00000000000000000023, 0.00000000000000000002
      )

def gammar(x):
    y = x - 1.0
    sm = _a[-1]
    for an in _a[-2::-1]:
        sm = sm * y + an
    return 1.0 / sm


# -------------------------------- exact solution ------------------------------------#
def exact_solution(x, y):
    return y**3 * (np.sin(x))**2


# ------------------------------ Neural Network Setup --------------------------------#
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------ Build FPDE -------------------------------------#
u = MLP()

def interior(n=N):
    x = torch.arange(delta, 1 + delta, T/n).reshape(-1, 1)
    y = torch.arange(delta, 1 + delta, T/n).reshape(-1, 1)

    coeff1 = torch.exp(-y)
    coeff2 = torch.sin(pi * x)
    coeff3 = pi**2 * torch.sin(pi * x)
    cond = -coeff1 * (coeff2 - coeff3)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def IC0(n=5):
    x = torch.rand(n, 1) * 2 - 1
    y = torch.zeros_like(x)
    cond = torch.sin(pi * x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def IC1(n=5):
    x = torch.rand(n, 1) * 2 - 1
    y = torch.ones_like(x)
    cond = torch.exp(-y) * torch.sin(pi * x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def BC_left(n=5):
    y = torch.rand(n, 1)
    x = -1 * torch.ones_like(y)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def BC_right(n=5):
    y = torch.rand(n, 1)
    x = torch.ones_like(y)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


loss = torch.nn.MSELoss()

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def caputo_dt(u, x, y, alpha):
    X_data = torch.cat([x, y], dim=1)
    Caputo_d_alpha_t = torch.tensor([])

    for tmp in X_data:
        x_tmp = tmp[0]
        y_tmp = tmp[1]
        sigma = 0
        k = int(y_tmp.detach().numpy() / delta)
        '''
        k = 0  Caputo = 0
        k = 1  Caputo = u - u0
        k = 2  Caputo = (b0 - b1) * u(x,t_j) = (1 - (2**(1 - alpha) - 1)) * u = 2 - 2**(1 - alpha)
        '''
        if k < 1:
            zero = torch.tensor([0])
            Caputo_d_alpha_t = torch.cat((Caputo_d_alpha_t, zero))

        elif k==1:
            du_x = (1 / gammar(2 - alpha) / delta ** alpha) * u(tmp)
            Caputo_d_alpha_t = torch.cat((Caputo_d_alpha_t, du_x))

        elif k==2:
            db = 2 - 2**(1 - alpha)
            y_k2 = torch.tensor([delta])
            x_tmp = torch.tensor([x_tmp])
            du_x = (1 / gammar(2 - alpha) / delta ** alpha) * (u(tmp) - db * u(torch.cat([x_tmp, y_k2])))
            tensor_copy = du_x.detach().clone()
            Caputo_d_alpha_t = torch.cat((Caputo_d_alpha_t, tensor_copy))

        else:
            for i in range(1, k-1):
                db = 2 * (k - i) ** (1 - alpha) - (k - i - 1) ** (1 - alpha) - (k - i + 1) ** (1 - alpha)
                t = delta * i
                y_k = torch.tensor([t])
                x_tmp = torch.tensor([x_tmp])
                sigma = sigma + db * u(torch.cat([x_tmp, y_k]))
            last_db = 2 - 2**(1 - alpha)
            y_last = torch.tensor([delta * (k - 1)])
            sigma = sigma + last_db * u(torch.cat([x_tmp, y_last]))
            du_x = (1 / gammar(2 - alpha) / delta ** alpha) * (u(tmp) - sigma)
            Caputo_d_alpha_t = torch.cat((Caputo_d_alpha_t, du_x))

    Caputo_d_alpha_t = Caputo_d_alpha_t.unsqueeze(1)
    return Caputo_d_alpha_t


def l_interior(u):
    x, y, cond = interior()
    uxy = u(torch.cat([x, y], dim=1))
    d_alpha_t = caputo_dt(u, x, y, alpha)
    dxx = gradients(uxy, x, 2)
    C = 0.9

    return loss(d_alpha_t - C * dxx, -cond)


def l_IC0(u):
    x, y, cond = IC0()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_IC1(u):
    x, y, cond = IC1()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_BC_left(u):
    x, y, cond = BC_left()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_BC_right(u):
    x, y, cond = BC_right()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


# ------------------------------ Training model -----------------------------------------------#
n_epochs = 15000
opt = torch.optim.Adam(params=u.parameters(), lr=1e-3)

X_LAB = []
losses = []
loss_IC0 = []
loss_IC1 = []
loss_BC1 = []
loss_BC2 = []
loss_PDE = []
for i in range(n_epochs):
    opt.zero_grad()
    l1 = l_interior(u)
    l2 = l_IC0(u)
    l3 = l_IC1(u)
    l4 = l_BC_left(u)
    l5 = l_BC_right(u)
    l = l1 + l2 + l3 + l4 + l5
    l.backward()
    opt.step()
    if i % 1000 == 0:
        X_LAB.append(i)
        losses.append(l.detach().cpu().numpy())
        loss_IC0.append(l2.detach().cpu().numpy())
        loss_IC1.append(l3.detach().cpu().numpy())
        loss_BC1.append(l4.detach().cpu().numpy())
        loss_BC2.append(l5.detach().cpu().numpy())
        loss_PDE.append(l1.detach().cpu().numpy())
        print("epochs %d  loss_sum %f  loss_ic0 %f  loss_ic1 %f  loss_bc1 %f  loss_bc2 %f  loss_pde %f" % (i, l, l2, l3, l4, l5, l1))


# ----------------------------------- plotting functions  ----------------------------------------------------#
plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.semilogy(X_LAB, losses, marker='*', linewidth='2', label="losses")
plt.semilogy(X_LAB, loss_IC0, marker='o', linewidth='2', label="loss_IC")
plt.semilogy(X_LAB, loss_IC1, marker='d', linewidth='2', label="loss_IC")
plt.semilogy(X_LAB, loss_BC1, marker='v', linewidth='2', label="loss_BC1")
plt.semilogy(X_LAB, loss_BC2, marker='+', linewidth='2', label="loss_BC2")
plt.semilogy(X_LAB, loss_PDE, marker='x', linewidth='2', label="loss_PDE")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.savefig("loss_new15000.png", dpi=300)
plt.show()