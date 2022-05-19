from parametrs_variation_funcs import Circuit
from parametrs_variation_funcs import main
import numpy as np
from random import*
number_of_circuits = 3
arguments = np.array([[15, -0.1, 100, 0.6, 1.0, 0.00128, 0.5, 2*np.sqrt(10**(-6)*choice([.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35])/np.pi)],
                [18, 0.8, 100, 0.4, 1.0, 10.0003, 4.5, 2*np.sqrt(10**(-6)*choice([.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35])/np.pi)],
            [13, 0.8, 10, 0.6, 0.4, 0.003000000000000001, 9.5, 2*np.sqrt(10**(-6)*choice([.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35])/np.pi)]])

circuits = [Circuit(*[15, -0.1, 100, 0.6, 1.0, 0.00128, 0.5, 2*np.sqrt(10**(-6)*choice([.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35])/np.pi)]),
            Circuit(*[18, 0.8, 100, 0.4, 1.0, 10.0003, 4.5, 2*np.sqrt(10**(-6)*choice([.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35])/np.pi)]),
            Circuit(*[13, 0.8, 10, 0.6, 0.4, 0.003000000000000001, 9.5, 2*np.sqrt(10**(-6)*choice([.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35])/np.pi)])]


limits = [0.64258981, 11.71964023]
b_min = [0, -1, 10**(-6), 0, 0, 0.00001, 0.5, 2*np.sqrt(10**(-6)*0.5/np.pi)]
b_max = [20, 1, 10**(3) , 1, 1, 10**3  , 10, 2*np.sqrt(10**(-6)*35/np.pi)]
bounds_min = np.array([b_min for i in range(number_of_circuits)])
bounds_max = np.array([b_max for i in range(number_of_circuits)])






Delta = arguments/10
A, B, C, D = main(circuits, number_of_circuits)
v_max = C[-1]

def create_circuits(args):
    Circuits = []
    for i in range(number_of_circuits):
        Circuits.append(Circuit(*args[i]))
    return Circuits


while np.linalg.norm(Delta)>0.0000001:
    delta_vector = np.zeros([3, 8])
    for i in range(number_of_circuits):
        for j in range(8):
            for k in range(2):

                charact = np.zeros([3, 8])
                charact[i, j] = (-1)**k
                new_args = Delta*charact+arguments
                new_circuits = create_circuits(new_args)

                A, B, C, D = main(new_circuits, number_of_circuits)

                for l in range(number_of_circuits):
                    Area = np.pi * (circuits[l].d / 2) ** 2
                    if max(abs(np.array(D[l]))) > np.exp(limits[1]) * Area ** limits[0]:
                        flag = True
                        continue
                for circ1 in new_circuits






