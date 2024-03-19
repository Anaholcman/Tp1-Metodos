import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.special import expit
from math import sin

# Función dada
def f(x):
    return (0.3) ** np.abs(x) * np.sin(4 * x) - expit(2 * x) + 2

# Generar puntos de datos en el intervalo [-4, 4]
x_data = np.linspace(-4, 4, num=9)  # Usamos 9 puntos equidistantes
y_data = f(x_data)

# Interpolar los puntos de datos con un polinomio de Lagrange
poly = lagrange(x_data, y_data)

# Crear un conjunto de puntos para graficar el polinomio interpolado
x_values = np.linspace(-4, 4, 1000)
y_values_interpolated = poly(x_values)

# Calcular los valores de la función original en los mismos puntos
y_values_original = f(x_values)

# Graficar la función original y el polinomio interpolado
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values_original, label='Función Original', color='blue')
plt.plot(x_values, y_values_interpolated, label='Polinomio Interpolado', linestyle='--', color='red')
plt.scatter(x_data, y_data, label='Puntos de Datos', color='black')
plt.title('Interpolación de Lagrange')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
