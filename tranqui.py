import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Función dada
def f(x):
    return (0.3) ** np.abs(x) * np.sin(4 * x) - np.tanh(2 * x) + 2

# Generar puntos de datos en el intervalo [-4, 4]
x_data = np.linspace(-4, 4, num=30)  # Usamos 9 puntos equidistantes
y_data = f(x_data)

# Lista para almacenar los polinomios de Lagrange lineales
polys = []

# Interpolar linealmente entre los puntos de datos
for i in range(len(x_data) - 1):
    poly = lagrange([x_data[i], x_data[i+1]], [y_data[i], y_data[i+1]])
    polys.append(poly)

# Crear un conjunto de puntos para graficar las interpolaciones lineales
x_values = np.linspace(-4, 4, 1000)
y_values_interpolated = np.zeros_like(x_values)

# Evaluar cada polinomio de Lagrange en su intervalo correspondiente
for i, poly in enumerate(polys):
    mask = (x_values >= x_data[i]) & (x_values <= x_data[i+1])
    y_values_interpolated[mask] = poly(x_values[mask])

# Graficar la función original y las interpolaciones lineales
plt.figure(figsize=(10, 6))
plt.plot(x_values, f(x_values), label='Función Original', color='blue')
plt.plot(x_values, y_values_interpolated, label='Interpolación Lineal de Lagrange', linestyle='--', color='red')
plt.scatter(x_data, y_data, label='Puntos de Datos', color='black')
plt.title('Interpolación Lineal de Lagrange')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
