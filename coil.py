import numpy as np
from math import e, pi, cos, sin


m = 0.001  # масса шара
p = 0.07 # Дж/Тл
mu0 = 1.25663706 * 10 ** -6  # магнитная константа

S = 1 # мм^2

rho = 0.0175*10**(-6)


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
        self.L = mu0 * self.n**2 * (x2 - x1) * pi * D**2
        #print("Индуктивность схема:", self.L)
        self.C = C
        self.R_add = R
        self.R = R#+rho*(x2-x1)/(np.pi*(d/2)**2)
        self.x1 = x1
        self.x2 = x2
        self.t0 = -1
    def __str__(self):
        return str([self.U0, self.x0, self.C, self.x1, self.x2, self.R_add, self.D, self.d])

def main(circuits, n_circuits):
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
                circuit.t0 = t # ОЧЕНЬ ВАЖНО ПОСЛЕ ВСЕГО ПОМЕНЯТЬ НА t
            gamma = circuit.R / (2 * circuit.L)
            omega0 = 1 / (circuit.L * circuit.C) ** 0.5
            if omega0 > gamma:
                #print("Живем в периодическом режиме")
                omega = (omega0 ** 2 - gamma ** 2) ** 0.5
                # print(circuit.C * circuit.U0 * e ** (-gamma * t) * (-gamma * np.cos(omega * (t - circuit.t0))))
                return circuit.C * circuit.U0 * e ** (-gamma * (t- circuit.t0)) * (
                        -gamma * np.cos(omega * (t - circuit.t0)) + omega * np.sin(omega * (t - circuit.t0)))
            elif omega0 == gamma:
                return circuit.C * circuit.U0 * gamma ** 2 * (t - circuit.t0) * e ** (-gamma * (t - circuit.t0))
            else:
                #print("Живем в апериодическом режиме")
                epsilon = (- omega0 ** 2 + gamma ** 2) ** 0.5

                try:
                    return circuit.C * circuit.U0 * (epsilon ** 2 - gamma ** 2) / (2 * epsilon) * e**(-(t - circuit.t0) * gamma) * (
                                   e**((t - circuit.t0) * epsilon) - e**(-((t - circuit.t0) * epsilon)))
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
        x_new = x - (circuit.x2 + circuit.x1)/2  # координата шарика относительно катушки
        #print(I, n, length, D, x, x_new)

        return mu0 / 2 * I * n * 8 * D ** 2 * (
                1 / ((length + 2 * x_new) ** 2 + 4 * D ** 2) ** (3 / 2) - 1 / ((length - 2 * x_new) ** 2 + 4 * D ** 2) ** (
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


    # n_circuit
    #  circuits
    #circuits = [Circuit(U0=24, x0=-0.05, C=10 ** (-5), R=0.1, D=0.001, x1=-0.05, x2=0.05), ]
    #circuits = [Circuit(U0=24, x0=-0.05, C=10 ** (-3), R=0.0001, D=0.001, x1=-0.05, x2=0.05), ]
    # #n_circuit = 1
    # circuits = [Circuit(U0=24, x0=-0.05, C=10 ** (-3), R=0.0001, D=0.001, x1=-0.05, x2=0.05), Circuit(U0=24, x0=0.1, C=10 ** (-3), R=0.0001, D=0.001, x1=0.1, x2=0.2)]
    # n_circuit = 2



    def current_for_plot(t, vars, circuit):
        """тестовая функция чтобы было удобно плотить. поменяли только t на 0 чтобы работало"""
        x = vars[0]
        if x >= circuit.x0  or circuit.t0 != -1:
            if circuit.t0 == -1:
                circuit.t0 = 0 # ОЧЕНЬ ВАЖНО ПОСЛЕ ВСЕГО ПОМЕНЯТЬ НА t
            gamma = circuit.R / (2 * circuit.L)
            omega0 = 1 / (circuit.L * circuit.C)**0.5
            if omega0 > gamma:
                omega = 1 / (omega0**2 - gamma ** 2) ** 0.5
           # print(circuit.C * circuit.U0 * e ** (-gamma * t) * (-gamma * np.cos(omega * (t - circuit.t0))))
                return -circuit.C * circuit.U0 * e ** (-gamma * t) * (
                        -gamma * np.cos(omega * (t - circuit.t0)) + omega * np.sin(omega * (t - circuit.t0)))
            if omega0 == gamma:
                return circuit.C * circuit.U0 * gamma**2 * (t - circuit.t0) * e ** (-gamma * (t - circuit.t0))
            else:
                epsilon = (- omega0**2 + gamma ** 2) ** 0.5
                return circuit.C * circuit.U0 * (epsilon**2 - gamma**2)/(2*epsilon) * (e**(-(t - circuit.t0)*gamma)) * (e**((t - circuit.t0)*epsilon) - e**(-((t - circuit.t0)*epsilon)))


        return 0


#def main():
    """n_circuit - количество RLC цепей
        circuits- массив с параметрами RLC цепей"""
    # время движения
    tmax = 3
    import numpy as np
    from scipy.integrate import ode
    ODE = ode(f)  # тут есть f1 и f. f написано к   ривыми ручками и рассчитано на бумажке f1 посчитал вольфрам
    ODE.set_integrator('dopri5', max_step=0.001, nsteps=500000000, safety=False)
    ODE.set_solout(step_handler)
    # circuits = [Circuit(U0=24, x0=-0.05, C=10 ** (-3), R=0.0001, D=0.001, x1=-0.05, x2=0.05),
    #             Circuit(U0=24, x0=0.1, C=10 ** (-3), R=0.0001, D=0.001, x1=0.1, x2=0.2)]
    # n_circuit = 2

    ODE.set_initial_value(np.array([-0.05, 0]), 0)  # задание начальных значений
    ODE.integrate(tmax)  # решение ОДУ

    #print(xs)
    #print(vs)
    #print(ts)
    #return ts, xs, vs, Is

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4)
    fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(ts[:700], xs[:700])
    # координата от времени
    axs[0].plot(ts, xs)
    #axs[0].set_xlim([0, 5])
    # axs[0].title("x(t)")
    # axs[1].plot(ts[1530:1550], vs[1530:1550])
    # скорость от времени
    axs[0].set_title(r"Координата $x$ от времени $t$")
    axs[1].plot(ts, vs)
    axs[1].set_title(r"Скорость $v$ от времени $t$")
    # ток от времени
    axs[2].plot(ts, Is[0])
    axs[2].set_title(r"Ток $I$ в цепи №1 от времени $t$")
    axs[3].plot(ts, Is[1])
    axs[3].set_title(r"Ток $I$ в цепи №2 от времени $t$")
    #axs[1].set_xlim([0, 5])
    # axs[1].title("v(t)")

    import pygame
    import os
    import math

    WIDTH = 1500
    HEIGHT = 720
    FPS = 30
    all_sprites = pygame.sprite.Group()

    # Создаем игру и окно
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Coil")
    clock = pygame.time.Clock()

    screen.fill((255, 255, 255))

    def load_image(name, color_key=None):
        fullname = os.path.join('', name)
        image = pygame.image.load(fullname).convert()
        if color_key is not None:
            if color_key == -1:
                color_key = image.get_at((0, 0))
            image.set_colorkey(color_key)
        else:
            image = image.convert_alpha()
        return image

    class Coil(pygame.sprite.Sprite):
        image_1, image_2 = [], []
        im = load_image("L/L0.png", -1)
        for i in range(51):
            im = load_image("L/L{n}.png".format(n= i), -1)
            im.set_colorkey((255, 255, 255))
            image_1.append(im)
        image_2.append(im)
        for i in range(1, 51):
            im = load_image("L/L-{n}.png".format(n= i), -1)
            im.set_colorkey((255, 255, 255))
            image_2.append(im)

        def __init__(self, circ):
            super().__init__(all_sprites)
            self.image = Coil.image_1[0]
            self.rect = Coil.image_1[0].get_rect()

            self.rect.x = circ.x1 * 1000 + 100
            self.rect.centery = 520

            self.I1 = []
            self.I2 = []

            for i in Coil.image_1:
                self.I1.append(pygame.transform.scale(i, ((circ.x2 - circ.x1) * 500, circ.D * 500)))
            for i in Coil.image_2:
                self.I2.append(pygame.transform.scale(i, ((circ.x2 - circ.x1) * 500, circ.D * 500)))

            self.Imax = 10
            self.I = 0

        def update(self, I):
            if self.Imax == 0:
                return None
            if abs(math.floor(51 * (I / self.Imax))) < 51:
                if I > 0:
                    self.image = self.I1[abs(math.floor(51 * I / self.Imax))]
                else:
                    self.image = self.I2[abs(math.floor(51 * I / self.Imax))]
            else:
                if I > 0:
                    self.image = self.I1[-1]
                else:
                    self.image = self.I2[-1]

    class Bullet(pygame.sprite.Sprite):
        image = load_image("bullet.png", -1)
        image.set_colorkey((255, 255, 255))
        image = pygame.transform.scale(image, (image.get_width() // 10, image.get_height() // 10))

        def __init__(self, x, y):
            super().__init__(all_sprites)
            self.rect = self.image.get_rect()

            self.rect.centerx = x
            self.rect.centery = y

            self.image = Bullet.image

        def update(self, x):
            self.rect.centerx = x + 100

    running = True

    L = []
    b = Bullet(0, 360)
    all_sprites.add(b)
    for i in range(n_circuits):
        z = Coil(circuits[i])
        z.Imax = max(Is[i])
        all_sprites.add(z)
        L.append(z)
    for i in range(len(ts)):
        screen.fill((255, 255, 255))
        for j in range(n_circuits):
            L[j].update(Is[j][i])
        b.update(xs[i] * 500)
        all_sprites.draw(screen)
        pygame.display.flip()
    pygame.quit()

    x = np.arange(-2, 2, 0.01)
    # plt.plot(x, f1(0, [x, 0])[1])
    plt.show()
    #return ts, xs, vs, Is
    return ts, xs, vs, Is


if __name__ == "__main__":
    n_circuit = 2
    def create_circuits(args):
        Circuits = []
        for i in range(n_circuit):
            Circuits.append(Circuit(*args[i]))
        return Circuits
    good_circuit = np.array([[1.338185915960346, -0.5606662736602739, 2.8716897855321095,
                              0.03943118032813586, 0.692536194926218, 0.03224658060051818,
                              0.07032583793884348, 0.0009772050238058398],
                             [0.9712912929804354 + 0.7, -0.4758710440494831, 11.96756807743066,
                              0.22080862076389352 + 0.7, 0.8729202340593378 + 0.7, 0.005204747565093779,
                              0.06373397580139444, 0.003568248232305542]])
    circuits = create_circuits(good_circuit)
    a, b, c, d = main(circuits, n_circuit)

    print(c[-1])
