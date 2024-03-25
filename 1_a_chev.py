import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline



def f_original(x):
    return (0.3) ** np.abs(x) * np.sin(4 * x) - np.tanh(2 * x) + 2


def chebyshev_points(a, b, n):
    k = np.arange(1, n+1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) / (2 * n) * np.pi)

a, b = -4, 4 
n = 15 # Número de puntos de Chebyshev

x_cheb = chebyshev_points(a, b, n)
y_cheb = f_original(x_cheb)

poly = lagrange(x_cheb, y_cheb) #lagrange chebyshev

# Puntos para graficar la función original
x_chev_tograf = np.linspace(a, b, 1000)
y_original = f_original(x_chev_tograf)
y_interpolated_bychev = poly(x_chev_tograf) #cheb


def absolute_error(original_values, interpolated_values):
    return np.abs(original_values - interpolated_values)

def relative_error(original_values, interpolated_values):
    return np.abs((original_values - interpolated_values) / original_values)

def calculate_errors(original_values, interpolated_values):
    abs_err = absolute_error(original_values, interpolated_values)
    rel_err = relative_error(original_values, interpolated_values)
    
    max_abs_err = np.max(abs_err)
    max_rel_err = np.max(rel_err)
    
    avg_abs_err = np.mean(abs_err) #devuelve el prom
    avg_rel_err = np.mean(rel_err)
    
    return texto_error(max_abs_err, max_rel_err, avg_abs_err, avg_rel_err)



def texto_error(max_abs_err, max_rel_err, avg_abs_err, avg_rel_err):
    return f"Error Absoluto Máximo: {max_abs_err:.5f}\n"\
             f"Error Absoluto Promedio: {avg_abs_err:.5f}\n\n"\
             f"Error Relativo Máximo: {max_rel_err:.5f}\n"\
             f"Error Relativo Promedio: {avg_rel_err:.5f}"


##graf chebyshev
fig = plt.figure(figsize=(15, 8))

ax1. = fig.add_subplot(3, 2, 1)




ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(x_chev_tograf, y_original, label='Función Original')
ax2.plot(x_chev_tograf, y_interpolated_bychev, label='Interpolación', linestyle='--', color='red')
ax2.plot(x_cheb, y_cheb, 'ro', label='Puntos de Chebyshev', color='black')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Interpolación con puntos de Chebyshev')
ax2.text(a, np.min(y_original), error_text_chevbyshev, fontsize=8, va='bottom', ha='left', color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax2.legend()
ax2.grid(True)
