# получаем уравнение движения магнита в катушке
# уравнение x'' = F/m

# Прости Дени но тут пока в СГС

m = 0.01 # масса в граммах
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
tmax = 10

# подрубаем scipy для решения диффура
import numpy as np
from scipy.integrate import ode
ODE = ode(f)
ODE.set_integrator('dopri5', max_step=0.0001)
ODE.set_solout(step_handler)

ODE.set_initial_value(np.array([0, 0]), 0)  # задание начальных значений
ODE.integrate(tmax)  # решение ОДУ

print(xs)
print(ts)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(ts, xs)
axs[1].plot(ts, vs)
plt.show()





