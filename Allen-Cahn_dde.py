import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import tf
import math

alpha = 0.9
N = 1000
T = 1
delta = T/N

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


def gamma(x):
    y = x - 1.0
    sm = _a[-1]
    for an in _a[-2::-1]:
        sm = sm * y + an
    return 1.0 / sm


def discrete_Caputo(x, y):
    sigma = 0
    for i in range(N):
        db = 2 * (N - i)**(1 - alpha) \
             - (N - i - 1)**(1 - alpha) \
             - (N - i + 1)**(1 - alpha)
        sigma = sigma + db * y

    coeff = (1 / gamma(2 - alpha) / delta**alpha)
    dy_t = coeff * \
           (y - sigma - tf.cos(math.pi * x[:, 0:1]) * x[:, 0:1]**2 * (N**(1 - alpha) - (N - 1)**(1 - alpha)))
    return dy_t


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return discrete_Caputo(x, y) - 0.0001 * dy_xx + 5 * y**3 - 5 * y


geom = dde.geometry.Hypercube(np.ones(19)*(-1),np.ones(19))
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(
    geomtime, lambda x: 0, lambda _, on_initial: on_initial
)
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
)

net = dde.maps.FNN([20] + [100] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)


losshistory, train_state = model.train(epochs=15000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

