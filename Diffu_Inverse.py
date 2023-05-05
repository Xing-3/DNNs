import deepxde as dde
import numpy as np
from deepxde.backend import tf
import math
import re
import matplotlib.pyplot as plt

alpha = 0.3
N = 1000
T = 1
delta = T/N
C = dde.Variable(2.0)

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


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


def pde(x, y):
    sigma = 0
    for i in range(1,N):
        db1 = 2 * (N - i)**(1 - alpha)
        db2 = (N - i - 1)**(1 - alpha)
        db3 = (N - i + 1)**(1 - alpha)
        db = db1 - db2 - db3
        sigma = sigma + db * y
    coeff = (1 / gamma(2 - alpha) / delta**alpha)
    dy_t = coeff * (y - sigma - tf.sin(math.pi * x[:, 0:1])
                    * (N**(1 - alpha) - (N - 1)**(1 - alpha)))
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return (
        dy_t
        - C * dy_xx
        + tf.exp(-x[:, 1:])
        * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
    )

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x), component=0)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    num_domain=80,
    num_boundary=20,
    num_initial=10,
    anchors=observe_x,
    solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile(
    "adam", lr=0.001, external_trainable_variables=C
)
variable = dde.callbacks.VariableValue(C, period=1000, filename="variables.dat")
losshistory, train_state = model.train(epochs=15000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

lines = open("variables.dat", "r").readlines()
vkinfer = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = vkinfer.shape
C_true = 0.9

plt.figure()
plt.plot(
    range(0, 1000 * l, 1000),
    np.ones(vkinfer[:, 0].shape) * C_true,
    color="black",
    label="Exact",
)
plt.plot(range(0, 1000 * l, 1000), vkinfer[:, 0], "b--", label="Pred")
plt.xlabel("Epoch")
plt.yscale("log")
plt.legend(frameon=False)
plt.ylabel("C")
plt.show()