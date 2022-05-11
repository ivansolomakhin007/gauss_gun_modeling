import numpy as np
from math import e, cos, sin

m = 0.01  # масса шара
mu0 = 1.25663706 * 10 ** -6  # магнитная константа


class Circuit:
    """U0 - напряжение конденсатора
       x0 - координата шара для запуска схемы
       C - емкость конденсатора
       L - индуктивность катушки
       x1 - начало катушки
       x2 - конец катушки
       R - сопротивление схемы
       D - радиус катушки
       t0 - храним время зажигания системы. None -> цепь еще не включилась
       """

    def __init__(self, U0, x0, C, L, R, D):
        self.U0 = U0
        self.x0 = x0
        self.L = L
        self.C = C
        self.R = R
        self.D = D
        self.t0 = None



# складируем времена, координаты и скорости
ts = []
xs = []
vs = []


def current(t, vars, circuit):
    """Ток теперь функция координаты"""
    x = vars[0]
    if x > circuit.x0:
        if not circuit.t0:
            circuit.t0 = t
        gamma = circuit.R / (2 * circuit.L)
        omega = (1/(circuit.L * circuit.C) - gamma**2)**0.5
        return circuit.C * circuit.U0 * e**(-gamma*t)*(-gamma*cos(omega*(t - circuit.t0)) + omega*sin(omega*(t - circuit.t0)))

def check_circuit():
    """на каждом шаге заставляем схему работать в зависимости от координаты активации и от координаты шара"""


def f(t, vars, *args):
    x = vars[0]

    working_circuits
    for circuit in circuits:

    return np.array([vars[1], mu0 / 2 * I * n * 8 * R ** 2 * (
                1 / ((L + 2 * x) ** 2 + 4 * R ** 2) ** (3 / 2) - 1 / ((L - 2 * x) ** 2 + 4 * R ** 2) ** (3 / 2))])

def check_circuit():
    """на каждом шаге заставляем схему работать"""
    return

def step_handler(t, vars):
    """Обработчик шага"""
    ts.append(t)
    xs.append(vars[0])
    vs.append(vars[1])


def main(n_circuit, circuits):
    """
        n_circuit - количество RLC цепей
        circuits- массив с параметрами RLC цепей
        """


main(1, [Circuit(U0=100, x0=5, C=10**(-9), L=10**(-1), R=5, D=0.02),])
