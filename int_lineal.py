import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange



def f_original(x):
    return (0.3) ** np.abs(x) * np.sin(4 * x) - np.tanh(2 * x) + 2

f_prima = np.diff(f_original)
print(f_prima)

def absolute_error(original_values, interpolated_values):
    return np.abs(original_values - interpolated_values)

def relative_error(original_values, interpolated_values):
    return np.abs((original_values - interpolated_values) / original_values)


def calculate_errors(original_values, interpolated_values):
    abs_err = absolute_error(original_values, interpolated_values)
    rel_err = relative_error(original_values, interpolated_values)
    
    max_abs_err = np.max(abs_err)
    max_rel_err = np.max(rel_err)
    
    avg_abs_err = np.mean(abs_err)
    avg_rel_err = np.mean(rel_err)
    
    return max_abs_err, max_rel_err, avg_abs_err, avg_rel_err



##### Interpolación lineal de Lagrange #####
x_data = np.linspace(-4, 4, num=15)  # Usamos 20 puntos equidistantes
y_data = f_original(x_data)



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

max_abs_err, max_rel_err, avg_abs_err, avg_rel_err = calculate_errors(f_original(x_values), y_values_interpolated)

error_text = f"Error Absoluto Máximo: {max_abs_err:.5f}\n"\
                f"Error Absoluto Promedio: {avg_abs_err:.5f}\n\n"\
                f"Error Relativo Máximo: {max_rel_err:.5f}\n"\
                f"Error Relativo Promedio: {avg_rel_err:.5f}"   


plt.plot()
    
# Graficar la función original y las interpolaciones lineales
plt.figure(figsize=(10, 6))
plt.plot(x_values, f_original(x_values), label='Función Original', color='blue')
plt.plot(x_values, y_values_interpolated, label='Interpolación Lineal de Lagrange', linestyle='--', color='red')
plt.scatter(x_data, y_data, label='Puntos de Datos', color='black')
plt.title('Interpolación Lineal de Lagrange')
plt.text(-4, np.min(f_original(x_values)), error_text, fontsize=8, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
