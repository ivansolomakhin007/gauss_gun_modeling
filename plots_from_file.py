from base_funcs import main, Circuit
import numpy as np
import matplotlib.pyplot as plt

n_circuit = 1

with open("data/vmax_v0=1.648930169304656.txt", "r") as f:
    lines = f.readlines()
    vmax = float(lines[-(n_circuit + 1)])
    print("Максимальная скорость: ", vmax)
    circs = []
    for i in range(n_circuit):
        print(lines[-1][1:-2])
        circ = np.array(list(map(float, lines[-1][1:-2].split(", "))))
        circs.append(circ)

v0 = 1.648930169304656

def create_circuits(args):
    Circuits = []
    for i in range(n_circuit):
        Circuits.append(Circuit(*args[i]))
    return Circuits

circuits = create_circuits(circs)
ts, xs, vs, Is = main(circuits, n_circuit, v0)

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
axs[2].plot(ts, Is[0])
# ток от времени
plt.show()

