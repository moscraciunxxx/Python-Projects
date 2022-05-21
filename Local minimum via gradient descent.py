# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 22:58:38 2022

@author: moscr
"""

# the function and its derivative

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
# from IPython import display
# display.set_matplotlib_formats('svg')  # make the graphs sharper
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')   # make the graphs sharper

def fx(x):

    return 3*x**2-3*x+4


def deriv(x):
    return 6*x-3


x = np.linspace(-2, 2, 2500)

plt.plot(x, fx(x), x, deriv(x))
plt.xlabel('x')
plt.ylabel('f(x)')

plt.legend(['f(x)', 'df'])
plt.grid('on')
plt.xlim(x[[0, -1]])
plt.show()


# # Plot Graph

# fig, ax1 = plt.subplots()
# ax1.plot(x, y)

# # Define Labels

# ax1.set_xlabel('X-axis')
# ax1.set_ylabel('Y1-axis')

# # Twin Axes

# ax2 = ax1.twinx()
# ax2.set_ylabel('Y2-axis')

# # Set limit

# plt.ylim(-1, 1)

# # Display

# plt.show()
# %% Gradient descent

# random starting point
localmin = np.random.choice(x, 1)

# GD parameters

learning_rate = .01
training_epoch = 100


# initialize the outputs matrix
modelparams = np.zeros((training_epoch, 2))


# training

for i in range(training_epoch):
    grad = deriv(localmin)
    localmin -= learning_rate*grad
    modelparams[i, :] = localmin, grad


plt.plot(x, fx(x), x, deriv(x))
plt.plot(localmin, fx(localmin), 'ro')
plt.plot(localmin, deriv(localmin), 'ro')

plt.xlabel('x')
plt.ylabel('f(x)')

plt.legend(['f(x)', 'df', 'local min'])
plt.grid('on')
plt.title('The empirical local min =%s ' % (f'{localmin[0]:.7}'))
plt.xlim(x[[0, -1]])
plt.show()


# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

for i in range(2):
    ax[i].plot(modelparams[:, i], 's-')
    ax[i].set_xlabel('Epoch')
    ax[i].set_title(f'Final estimated minimum: {localmin[0]:.7}')

ax[0].set_ylabel('Local minimum')
ax[1].set_ylabel('Gradient')


plt.show()


# %% Repeat in 2D

def peaks(x, y):
    x, y = np.meshgrid(x, y)

    z = 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)\
        - 1/3*np.exp(-(x+1)**2-y**2)

    return z


x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)


Z = peaks(x, y)

plt.imshow(Z, extent=[x[0], x[-1], y[0], y[-1]],
           origin='lower', vmin=-5, vmax=5)
plt.show()


sx, sy = sym.symbols('sx,sy')

sz = 3*(1-sx)**2*sym.exp(-(sx**2)-(sy+1)**2)-10*(sx/5-sx**3-sy**5)*sym.exp(-sx**2-sy**2)\
    - 1/3*sym.exp(-(sx+1)**2-sy**2)


# compute the symbolic derivative
df_x = sym.lambdify((sx, sy), sym.diff(sz, sx), 'sympy')
# compute the symbolic derivative
df_y = sym.lambdify((sx, sy), sym.diff(sz, sy), 'sympy')

df_x(1, 1).evalf()   # evaluat the derivative


# %% Grad Desc


# random starting point
localmin = np.random.rand(2)*4-2

# GD parameters

learning_rate = .01
training_epoch = 1000


# initialize the outputs matrix
modelparams = np.zeros((training_epoch, 2))


# training

for i in range(training_epoch):
    dx = df_x(localmin[0], localmin[1])
    dy = df_y(localmin[0], localmin[1])

    grad = np.array([dx, dy])
    localmin = localmin - learning_rate*grad   # broadcasting
    modelparams[i, :] = localmin


plt.imshow(Z, extent=[x[0], x[-1], y[0], y[-1]],
           origin='lower', vmin=-5, vmax=5)
plt.plot(modelparams[0, 0], modelparams[0, 1], 'bs')
plt.plot(modelparams[-1, 0], modelparams[-1, 1], 'ro')
plt.plot(modelparams[:, 0], modelparams[:, 1], 'm')
plt.show()


# Bonus video  3D

fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection='3d')


X, Y = np.meshgrid(x, y)

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=.3)
ax.scatter(modelparams[0, 0], modelparams[0, 1], peaks(modelparams[0, 0], modelparams[0, 1]), 'bs')

ax.scatter(modelparams[-1, 0], modelparams[-1,1],peaks(modelparams[-1,0],modelparams[-1,1]),'ro')


ax.view_init(40, 20)

ax.axis('off')
plt.show()






























