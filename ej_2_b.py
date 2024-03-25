import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.optimize import fsolve

data = np.loadtxt('mnyo_tp01_2024_1s/mnyo_mediciones.csv')
x_gtruth = data[:, 0]
y_truth = data[:, 1]

puntos_4 = np.loadtxt("mnyo_tp01_2024_1s/mnyo_mediciones2.csv")
x_4 = puntos_4[:, 0]
y_4 = puntos_4[:, 1]

interpolacion_4puntos = lagrange(x_4, y_4)
interpolacion_ground_truth = lagrange(x_gtruth, y_truth)

print(f" interpolacion 4puntos{interpolacion_4puntos}")

def y_poly(x):
    return interpolacion_4puntos(x)

##calcular e jacobiano de la funcion 4 puntos




plt.figure(figsize=(8, 6))
plt.plot(x_gtruth, y_truth, color='red', linestyle='-', marker='', label='Recorrido 1era trayectoria')
plt.plot(x_4, y_4, color='green', linestyle='-', marker='', label='Recorrido 2nda trayectoria')
plt.scatter(x_4, y_4, label='4 mediciones', color='orange')  # Utilizamos los datos (x_4, y_4) para scatter
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de las 2 trayectorias')
plt.legend()
plt.grid(True)
plt.show()
