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
    tmax = 1
    import numpy as np
    from scipy.integrate import ode
    ODE = ode(f)  # тут естьaxs[0].axhline(circuits[2].x1,color="red", ls="--") f1 и f. f написано кривыми ручками и рассчитано на бумажке f1 посчитал вольфрам
    ODE.set_integrator('dopri5', max_step=0.001, nsteps=700000)
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
    fig, axs = plt.subplots(3)
    fig.suptitle('Зависимости координаты диполя $x$, скорости $v$ и токов $I$ в каждой из цепи от времени $t$', fontsize=14)
    # axs[0].plot(ts[:700], xs[:700])
    # координата от времени
    axs[0].plot(ts, xs)
    #axs[0].set_xlim([0, 5])
    # axs[0].title("x(t)")
    # axs[1].plot(ts[1530:1550], vs[1530:1550])
    # скорость от времени
    #axs[0].set_title(r"Координата $x$ от времени $t$")
    axs[0].set_ylabel(r"$x$, м", fontsize=14)
    # axs[0].axhline(circuits[2].x1,color="red", ls="--")
    # axs[0].axhline(circuits[2].x2, color="red", ls="--")
    axs[1].plot(ts, vs)
    #axs[1].set_title(r"Скорость $v$ от времени $t$")
    axs[1].set_ylabel(r"$v$, м/c", fontsize=14)

    # ток от времени
    print(current(np.array(ts), [-0.05, 0], circuits[0]))
    print(max(Is[1]))
    axs[2].plot(ts, Is[2])
    axs[2].plot(ts, Is[1])
    axs[2].plot(ts, Is[0])
    axs[2].set_ylabel(r"$I$, А", fontsize=14)
    axs[2].set_xlabel(r"$t$, с", fontsize=14)
    #axs[2].plot(ts, Is[1])
    #axs[2].plot(ts, Is[2])
    #axs[2].set_title(r"Ток $I$ в цепях от времени $t$")
    axs[2].legend(["цепь №3", "цепь №2", "цепь №1"])
    # axs[3].plot(ts, Is[1])
    # axs[3].set_title(r"Ток $I$ в цепи №2 от времени $t$")
    # axs[4].plot(ts, Is[2])
    # axs[4].set_title(r"Ток $I$ в цепи №3 от времени $t$")
    #axs[1].set_xlim([0, 5])
    # axs[1].title("v(t)")

    x = np.arange(-2, 2, 0.01)
    # plt.plot(x, f1(0, [x, 0])[1])
    plt.show()
    #return ts, xs, vs, Is
    return ts, xs, vs, Is


if __name__ == "__main__":
    n_circuit = 3
    def create_circuits(args):
        Circuits = []
        for i in range(n_circuit):
            Circuits.append(Circuit(*args[i]))
        return Circuits
    good_circuit = np.array([[12.458233916180461, 0.6030454188449583, 0.10040643898725007, 0.39270207622388487, 0.6026158237331962, 0.3030899913787122, 0.046431740741472646, 0.001784124116152771],
[1.5692778946870756,-0.6173846265217682, 0.00012796282308287797, 0.903188183256714, 0.7026522669977449, 0.001059592317529516, 0.03748410297345381, 0.00451351666838205],
[5.0634897936052115, -0.7771950942245056, 0.19815932735064856, 0.010042233877513684, 0.8795643961507403, 0.03367287704474973, 0.0995586783271452, 0.0007978845608028653]])
    circuits = create_circuits(good_circuit)
    good_circuit = np.array([[7.816518129334556, 0.48771984029063176, 0.0007099733703299516, 0.1292375788275668, 0.35913975584079116, 0.4860687183243379, 0.04333893953982519, 0.001784124116152771],
[12.408415396142336, 0.8776678083403271, 0.0021049089092351797, 0.47965767722964225, 0.38004425422889476, 0.5319261038150805, 0.03973000676806667, 0.0009772050238058398],
[0.4847946822274829, -0.27281746000397233, 6.0492870107731, 0.022587208323198937, 0.5653580882154727, 0.005324506514993581, 0.07133918050389702, 0.001784124116152771]]) #vmax = 1.112401544463721
    circuits = create_circuits(good_circuit)
    good_circuit = np.array([[0.6859594398378199, -0.8176488570472205, 15.51181838154246, 0.0800250253683924, 0.5603124721568035, 0.007923288153583462, 0.08146330237931881, 0.0015957691216057306],
[11.685570272460591, 0.05363878976243752, 0.014629829875145403, 0.5815770205770463, 0.8683869325291719, 15.024156291235093, 0.04048643450414497, 0.0007978845608028653],
[14.009308298933334, 0.9664254529187757, 0.012371528784658773, 0.3016477463348878, 0.15784009334236715, 0.0028330907955161633, 0.02645549673804726, 0.002256758334191025]])  # vmax = 1.4888140586921594
    circuits = create_circuits(good_circuit)
    good_circuit = np.array([[20, -0.8176488570472205, 15.51181838154246, 0.0800250253683924,
                              0.5603124721568035, 0.007923288153583462, 0.08146330237931881, 0.0015957691216057306],
                             [20, 0.05363878976243752, 0.014629829875145403, 0.5815770205770463,
                              0.8683869325291719, 15.024156291235093, 0.04048643450414497, 0.0007978845608028653],
                             [20, 0.9664254529187757, 0.012371528784658773, 0.3016477463348878,
                              0.15784009334236715, 0.0028330907955161633, 0.02645549673804726,
                              0.002256758334191025]])  # vmax = 1.4888140586921594
    circuits = create_circuits(good_circuit)
#     good_circuit = np.array([[2.1658215629444566, 0.36710242019133843, 0.007755551990353316, 0.05020076578246202, 0.7300531578543034, 36.97109760679317, 0.04543103536452905, 0.006675581178124545],
# [2.6350425737266914, 0.912618412522167, 0.000611925901507773, 0.4632898504160191, 0.8980985544097955, 13.367566354619006, 0.03287848511037851, 0.002763953195770684],
# [0.5949255457166269, -0.35677927363524087, 3.2900264404250747, 0.06017557745610236, 0.38095221401903, 0.011453375822213847, 0.06570785792367652, 0.0009772050238058398]])
#     circuits = create_circuits(good_circuit)
    # circuits = [Circuit(U0=24, x0=-0.05, C=10 ** (-3), R=0.01, D=0.001, x1=-0.05, x2=0.05, d=0.002),
    #             Circuit(U0=24, x0=0.1, C=10 ** (-3), R=0.001, D=0.001, x1=0.1, x2=0.2, d=0.002)]
    # n_circuit = 2

    a, b, c, d = main(circuits, n_circuit)
    D = d
    flag = False
    limits = [0.64258981, 11.71964023]

    for l in range(3):
        Area = np.pi*(circuits[l].d/2)**2
        if max(abs(np.array(D[l])))>np.exp(limits[1])*Area**limits[0]:
            flag = True
    print(flag)
    print(a)
    print(b)
    print(c)
    print(d)
    print(c[-1])