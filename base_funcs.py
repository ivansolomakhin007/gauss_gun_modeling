import numpy as np
from math import e, pi, cos, sin

m = 0.001  # масса шара
p = 0.07  # Дж/Тл
mu0 = 1.25663706 * 10 ** -6  # магнитная константа

S = 1  # мм^2

rho = 0.0175 * 10 ** (-6)


class Circuit:
    """U0 - напряжение конденсатора
       x0 - координата шара для запуска схемы
       C - емкость конденсатора
       L - индуктивность катушки
       n -плотность намотки
       x1 - начало катушки
       x2 - конец катушки
       R - дополнительное сопротивление схемы
       D - радиус катушки
       d - диаметр провода
       t0 - храним время зажигания системы. None -> цепь еще не включилась
       """

    def __init__(self, U0, x0, C, x1, x2, R, D, d):
        self.U0 = U0
        self.x0 = x0
        self.d = d
        self.n = 1 / d
        self.D = D
        self.L = mu0 * self.n ** 2 * (x2 - x1) * pi * D ** 2
        # print("Индуктивность схема:", self.L)
        self.C = C
        self.R_add = R
        self.R = R  # +rho*(x2-x1)/(np.pi*(d/2)**2)
        self.x1 = x1
        self.x2 = x2
        self.t0 = -1

    def __str__(self):
        return str([self.U0, self.x0, self.C, self.x1, self.x2, self.R_add, self.D, self.d])


def main(circuits, n_circuits, v0):
    # складируем времена, координаты и скорости
    ts = []
    xs = []
    vs = []
    Is = []
    for i in range(n_circuits):
        Is.append([])

    # 2d array токов, iй схеме сопоставлен iй массив токов в каждый момент времени

    def current(t, vars, circuit):
        """Ток теперь функция координаты"""
        x = vars[0]
        if x >= circuit.x0 or circuit.t0 != -1:
            if circuit.t0 == -1:
                circuit.t0 = t  # ОЧЕНЬ ВАЖНО ПОСЛЕ ВСЕГО ПОМЕНЯТЬ НА t
            gamma = circuit.R / (2 * circuit.L)
            omega0 = 1 / (circuit.L * circuit.C) ** 0.5
            if omega0 > gamma:
                omega = (omega0 ** 2 - gamma ** 2) ** 0.5
                return circuit.C * circuit.U0 * e ** (-gamma * t) * (
                        -gamma * np.cos(omega * (t - circuit.t0)) + omega * np.sin(omega * (t - circuit.t0)))
            elif omega0 == gamma:
                return circuit.C * circuit.U0 * gamma ** 2 * (t - circuit.t0) * e ** (-gamma * (t - circuit.t0))
            else:
                epsilon = (- omega0 ** 2 + gamma ** 2) ** 0.5
                try:
                    return circuit.C * circuit.U0 * (epsilon ** 2 - gamma ** 2) / (2 * epsilon) * e ** (
                                -(t - circuit.t0) * gamma) * (
                                   e ** ((t - circuit.t0) * epsilon) - e ** (-((t - circuit.t0) * epsilon)))
                except OverflowError:
                    return 0

        return 0

    def check_circuit(t, vars, circuit):
        x = vars[0]
        if x >= circuit.x0:
            if circuit.t0 == -1:
                circuit.t0 = t

    def force_by_induct(t, vars, circuit):
        I = current(t, vars, circuit)
        n = circuit.n
        length = circuit.x2 - circuit.x1
        D = circuit.D
        x = vars[0]
        x_new = x - (circuit.x2 + circuit.x1) / 2  # координата шарика относительно катушки
        return mu0 / 2 * I * n * 8 * D ** 2 * (
                1 / ((length + 2 * x_new) ** 2 + 4 * D ** 2) ** (3 / 2) - 1 / (
                    (length - 2 * x_new) ** 2 + 4 * D ** 2) ** (
                        3 / 2)) * p / m

    def f(t, vars):
        x = vars[0]
        force = 0
        for circuit in circuits:
            check_circuit(t, vars, circuit)
            if circuit.t0 != -1:
                force += force_by_induct(t, vars, circuit)
        return np.array([vars[1], force])

    def step_handler(t, vars):
        """Обработчик шага"""
        ts.append(t)
        xs.append(vars[0])
        vs.append(vars[1])
        for i in range(n_circuits):
            Is[i].append(current(t, [vars[0], vars[1]], circuits[i]))

    # время движения
    tmax = 1
    import numpy as np
    from scipy.integrate import ode
    ODE = ode(f)  # тут есть f1 и f. f написано кривыми ручками и рассчитано на бумажке f1 посчитал вольфрам
    ODE.set_integrator('dopri5', max_step=0.001, nsteps=70000)
    ODE.set_solout(step_handler)
    ODE.set_initial_value(np.array([-0.05, v0]), 0)  # задание начальных значений
    ODE.integrate(tmax)  # решение ОДУ
    return ts, xs, vs, Is
