import numpy as np
from math import e, pi, cos, sin

m = 0.001  # масса шара
p = 0.07 # Дж/Тл
mu0 = 1.25663706 * 10 ** -6  # магнитная константа

S = 1 # мм^2
d = 2*(S/pi)**0.5 * 10**-3 # м



class Circuit:
    """U0 - напряжение конденсатора
       x0 - координата шара для запуска схемы
       C - емкость конденсатора
       L - индуктивность катушки
       n -плотность намотки
       x1 - начало катушки
       x2 - конец катушки
       R - сопротивление схемы
       D - радиус катушки
       t0 - храним время зажигания системы. None -> цепь еще не включилась
       """

    def __init__(self, U0, x0, C, x1, x2, R, D):
        self.U0 = U0
        self.x0 = x0
        self.n = 1 / d
        self.D = D
        self.L = mu0 * self.n**2 * (x2 - x1) * pi * D**2
        print("Индуктивность схема:", self.L)
        self.C = C
        self.R = R
        self.x1 = x1
        self.x2 = x2
        self.t0 = None


# складируем времена, координаты и скорости
ts = []
xs = []
vs = []


def current(t, vars, circuit):
    """Ток теперь функция координаты"""
    x = vars[0]
    if x >= circuit.x0:
        if not circuit.t0:
            circuit.t0 = t # ОЧЕНЬ ВАЖНО ПОСЛЕ ВСЕГО ПОМЕНЯТЬ НА t
        gamma = circuit.R / (2 * circuit.L)
        omega = (1 / (circuit.L * circuit.C) - gamma ** 2) ** 0.5
       # print(circuit.C * circuit.U0 * e ** (-gamma * t) * (-gamma * np.cos(omega * (t - circuit.t0))))
        return -circuit.C * circuit.U0 * e ** (-gamma * t) * (
                    -gamma * np.cos(omega * (t - circuit.t0)) + omega * np.sin(omega * (t - circuit.t0)))
    return 0


def check_circuit(t, vars, circuit):
    x = vars[0]
    if x >= circuit.x0:
        if not circuit.t0:
            circuit.t0 = t



def force_by_induct(t, vars, circuit):
    I = current(t, vars, circuit)
    n = circuit.n
    length = circuit.x2 - circuit.x1
    D = circuit.D
    x = vars[0]
    x_new = x - (circuit.x2 + circuit.x1)  # координата шарика относительно катушки
    #print(I, n, length, D, x, x_new)

    return mu0 / 2 * I * n * 8 * D ** 2 * (
            1 / ((length + 2 * x_new) ** 2 + 4 * D ** 2) ** (3 / 2) - 1 / ((length - 2 * x_new) ** 2 + 4 * D ** 2) ** (
                3 / 2)) * p / m


def f(t, vars):
    x = vars[0]
    force = 0
    for circuit in circuits:
        check_circuit(t, vars, circuit)
        if circuit.t0:
            force += force_by_induct(t, vars, circuit)
    return np.array([vars[1], force])




def step_handler(t, vars):
    """Обработчик шага"""
    ts.append(t)
    xs.append(vars[0])
    vs.append(vars[1])


# n_circuit
#  circuits
circuits = [Circuit(U0=24, x0=-0.05, C=10 ** (-3), R=0.01, D=0.001, x1=-0.05, x2=0.05), ]
n_circuit = 1



def main():
    """
        n_circuit - количество RLC цепей
        circuits- массив с параметрами RLC цепей
        """
    # время движения
    tmax = 0.0001
    import numpy as np
    from scipy.integrate import ode
    ODE = ode(f)  # тут есть f1 и f. f написано кривыми ручками и рассчитано на бумажке f1 посчитал вольфрам
    ODE.set_integrator('dopri5', max_step=0.001, nsteps=70000)
    ODE.set_solout(step_handler)

    ODE.set_initial_value(np.array([-0.05, 0]), 0)  # задание начальных значений
    ODE.integrate(tmax)  # решение ОДУ

    print(xs)
    print(vs)
    print(ts)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(ts[:700], xs[:700])
    axs[0].plot(ts, xs)
    #axs[0].set_xlim([0, 5])
    # axs[0].title("x(t)")
    # axs[1].plot(ts[1530:1550], vs[1530:1550])
    axs[1].plot(ts, vs)
    #axs[1].set_xlim([0, 5])
    # axs[1].title("v(t)")

    x = np.arange(-2, 2, 0.01)
    # plt.plot(x, f1(0, [x, 0])[1])
    plt.show()


main()



def current_plot():
    t = np.arange(0.0, 0.1, 0.0001)
    import matplotlib.pyplot as plt
    #c1 = np.vectorize(current)
    print(current(t, np.array([-0.05, 0.0]), circuits[0]))
    plt.plot(t, current(t, [-0.05, 0], circuits[0]))
    plt.show()

#current_plot()