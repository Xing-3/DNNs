import numpy as np
import sciann as sn
from sciann import Variable, Functional, SciModel
from sciann.constraints import Data
from sciann.utils.math import sign, sin, diff
import math
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

'''
Burgers equation
pde: u_t + u*u_x - (0.01/pi)*u_xx = 0;  t in [0,1], x in [-1,1]
ic : u(0,x) = -sin(pi*x)
bc : u(t,-1) = u(t,1) = 0
'''

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

pi = np.pi
k = 500
T = 2
alpha = 0.3
delta = k / T
t = sn.Variable("t")
x = sn.Variable("x")
u = sn.Functional("u", [x, t], 2*[20], "tanh")

def discrete_Caputo(u, x, alpha):
    sigma = 0
    for j in range(k):
        db = 2 * (k - j) ** (1 - alpha) - (k - j - 1) ** (1 - alpha) - (k - j + 1) ** (1 - alpha)
        sigma = sigma + db * u
    dy_t = (1 / gammar(2 - alpha) / delta**alpha) * (u - sigma + sn.sin(math.pi * x) * (k**(1 - alpha) - (k - 1)**(1 - alpha)))
    return dy_t

TOL = 0.001
L1 = discrete_Caputo(u, x, alpha) + u * diff(u, x) - (0.01 / pi) * diff(u, x, order=2)
L2 = (1 - sign(t - TOL)) * (u + sin(pi*x))
L3 = (1 - sign(x - (-1 + TOL))) * u
L4 = (1 + sign(x - (1 - TOL))) * u

m = sn.SciModel([x, t], [L1, L2, L3, L4], optimizer='adam')

# data
x_data, t_data = np.meshgrid(
    np.linspace(-1, 1, 500),
    np.linspace(0, 1, 500)
) # 1000*2

h = m.train([x_data, t_data], 4*['zero'], learning_rate=0.001, epochs=5000, batch_size=200, adaptive_weights={'method': 'NTK', 'freq': 10, 'use_score': True})

plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


x_test, t_test = np.meshgrid(
    np.linspace(-1, 1, 200),
    np.linspace(0, 1, 200)
)
u_pred = u.eval(m, [x_test, t_test])
fig = plt.figure(figsize=(3, 4))
plt.pcolor(x_test, t_test, u_pred, cmap='seismic')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()
