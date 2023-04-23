import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from deepxde.backend import tf
import math

alpha = 0.9
N = 1000
T = 1
delta = T/N

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    sigma = 0
    for i in range(N):
        db = 2 * (N - i)**(1 - alpha) \
             - (N - i - 1)**(1 - alpha) \
             - (N - i + 1)**(1 - alpha)
        sigma = sigma + db * y
    dy_t = (1 / gamma(2 - alpha) / delta**alpha) * \
           (y - sigma - tf.cos(math.pi * x[:, 0:1]) * x[:, 0:1]**2 * (N**(1 - alpha) - (N - 1)**(1 - alpha)))

    return dy_t - 0.0001 * dy_xx + 5 * y**3 - 5 * y


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

