import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

data = np.loadtxt('mnyo_tp01_2024_1s/mnyo_mediciones.csv')
x = data[:, 0]
y = data[:, 1]

puntos_4 = np.loadtxt("mnyo_tp01_2024_1s/mnyo_mediciones2.csv")
x_4 = puntos_4[:, 0]
y_4 = puntos_4[:, 1]

poly = lagrange(x_4, y_4)

print(poly)

def y(x):
    return poly(x)








plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Puntos')
plt.plot(x, y, color='red', linestyle='-', marker='', label='Recorrido 1era trayectoria')
plt.plot(x_4, y_4, color='green', linestyle='-', marker='', label='Recorrido 2nda trayectoria')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x_4, y_4, label='4 mediciones', color='orange')
plt.title('Grafico de las 2 trayectorias')
plt.legend()
plt.grid(True)
plt.show()


