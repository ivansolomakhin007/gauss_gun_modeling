from parametrs_variation_funcs import Circuit
from parametrs_variation_funcs import main
import numpy as np
from random import*
All_possibilities = []
number_of_circuits = 3
N = 10**6
rho = 0.0175*10**(-6)
v_max = 1
kpd_max = 0
curcuits_max_kpd = []
circuits_max_vel = []
limits = [0.64258981, 11.71964023]
for i in range(N):
    circuits = []
    z = False
    all_x = []
    for tr in range(number_of_circuits):
        x1 = randint(0, 10)/10
        x2 = randint(0, 10)/10
    for j in range(number_of_circuits):
        u = randint(0, 20)
        c = 10**(randint(-6, 2))
        R = 10**(randint(-3, 2))
        radius = np.sqrt(10**(-6)*choice([.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35])/np.pi)
        width = randint(1, 20)/2
        x0 = randint(-10, 10)/10

        if x2<x1:
            z = True
            continue
        R_sum = R + rho*(x2-x1)/(np.pi*radius**2)
        circuits.append(Circuit(U0=u, x0=x0, C=c, R=R_sum, D=width, x1=x1, x2=x2, d = 2*radius))

    if z:
        continue
    h = False
    for circ1 in circuits:
        for circ2 in circuits:
            if circ1.x2 > circ2.x1:
                h = True
                continue
        if h:
            continue
    if h:
        continue
    if i%1000==0:
        print(i)
    A, B, C, D = main(circuits, number_of_circuits)
    flag = False
    for l in range(number_of_circuits):
        Area = np.pi*(circuits[l].d/2)**2
        if max(abs(np.array(D[l])))>np.exp(limits[1])*Area**limits[0]:
            flag = True
            continue
    if flag:
        continue
    print(i)
    Energy = 0
    for circ in circuits:
        Energy+=(circ.C*circ.U0**2)/2
    if C[-1]>v_max:
        v_max = C[-1]
        circuits_max_vel = circuits
        print('------new velosity maximum--------')
        print(v_max)
        for circs in circuits:
            print(circs)
        print('------------------------')
    if (0.001*(C[-1])**2/2)/(Energy)>kpd_max:
        kpd_max = C[-1]
        circuits_max_kpd = circuits
        print('------new velosity maximum--------')
        print(kpd_max)
        for circs in circuits:
            print(circs)
        print('------------------------')







