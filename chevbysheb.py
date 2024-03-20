import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Generar puntos de Chebyshev
def chebyshev_points(a, b, n):
    k = np.arange(1, n+1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) / (2 * n) * np.pi)

# Funcion original
def f_original(x):
    return (0.3) ** np.abs(x) * np.sin(4 * x) - np.tanh(2 * x) + 2

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


a, b = -4, 4 # Intervalo


n = 15 # Número de puntos de Chebyshev

# Generar puntos ¡
x_cheb = chebyshev_points(a, b, n)
y_cheb = f_original(x_cheb)


poly = lagrange(x_cheb, y_cheb)# Interpolación polinómica de Lagrange

# Puntos para graficar la función original
x = np.linspace(a, b, 1000)
y_original = f_original(x)
y_interpolated = poly(x)


max_abs_err, max_rel_err, avg_abs_err, avg_rel_err = calculate_errors(y_original, y_interpolated)# Calcular errores

error_text = f"Error Absoluto Máximo: {max_abs_err:.5f}\n"\
             f"Error Absoluto Promedio: {avg_abs_err:.5f}\n\n"\
             f"Error Relativo Máximo: {max_rel_err:.5f}\n"\
             f"Error Relativo Promedio: {avg_rel_err:.5f}"

# Graficar la función original y la interpolación
plt.plot(x, y_original, label='Función Original')
plt.plot(x, y_interpolated, label='Interpolación')
plt.plot(x_cheb, y_cheb, 'ro', label='Puntos de Chebyshev')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolación con puntos de Chebyshev')
plt.text(a, np.min(y_original), error_text, fontsize=12, va='bottom', ha='left', color='red')
plt.legend()
plt.grid(True)
plt.show()
