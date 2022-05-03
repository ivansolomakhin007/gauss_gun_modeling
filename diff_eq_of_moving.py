# получаем уравнение движения магнита в катушке
# уравнение x'' = F/m

# Прости Дени но тут пока в СГС

m = 0.01 # масса в килограммах
c = 3*10**10 # скорость света
mu0 = 1.25663706 * 10**-6

I = 30 # ток в катшке А
n = 500 # витков/м
L = 1 # длина катушки, м
R = 0.02 # радиус катушки, м

def f(t, vars):
    # vars = (x, x')
    x = vars[0]
    return np.array([vars[1], mu0/2 * I * n * ((L**2 / 4 + 3/2 *L*x - R**2 - 2*x**2)/((L/2 - x)**2 + R**2)**3/2 + (R**2 + L**2/4 + x*L/2)/
                                  ((L/2 + x)**2 + R**2)**3/2)])


def f1(t, vars):
    x = vars[0]
    return np.array([vars[1],  mu0/2 * I * n *8*R**2*(1/((L + 2*x)**2 + 4*R**2)**(3/2) - 1/((L - 2*x)**2 + 4*R**2)**(3/2))])

# складируем времена, координаты и скорости
ts = []
xs = []
vs = []

def step_handler(t, vars):
    """Обработчик шага"""
    ts.append(t)
    xs.append(vars[0])
    vs.append(vars[1])


# время движения
tmax = 40

def field(x):
    return mu0*I*n/2 * ((L/2 - x)/((L/2 - x)**2 + R**2) + (L/2 + x)/((L/2 + x)**2 + R**2))

from scipy.misc import derivative

def f_new(t, vars):
    x = vars[0]
    print(derivative(field, x, dx=10**-6))
    return np.array(vars[0], derivative(field, x, dx=10**-6))


# подрубаем scipy для решения диффура
import numpy as np
from scipy.integrate import ode
ODE = ode(f1) # тут есть f1 и f. f написано кривыми ручками и рассчитано на бумажке f1 посчитал вольфрам
ODE.set_integrator('dopri5', max_step=0.001, nsteps=70000)
ODE.set_solout(step_handler)

ODE.set_initial_value(np.array([-0.55, 0]), 0)  # задание начальных значений
ODE.integrate(tmax)  # решение ОДУ

print(xs)
print(ts)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
#axs[0].plot(ts[:700], xs[:700])
axs[0].plot(ts, xs)
#axs[0].title("x(t)")
#axs[1].plot(ts[1530:1550], vs[1530:1550])
axs[1].plot(ts, vs)
#axs[1].title("v(t)")

x = np.arange(-2, 2, 0.01)
#plt.plot(x, f1(0, [x, 0])[1])
plt.show()





