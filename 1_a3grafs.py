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

##interpolacion lineal de lagrange----------------------

x_data_lineal = np.linspace(-4, 4, num=15)  
y_data_lineal = f_original(x_data_lineal)

polys = []

for i in range(len(x_data_lineal) - 1):
    poly = lagrange([x_data_lineal[i], x_data_lineal[i+1]], [y_data_lineal[i], y_data_lineal[i+1]])
    polys.append(poly)

x_values_lineal = np.linspace(-4, 4, 1000)# conjunto de puntos para graficar las interpolaciones lineales
y_values_interpolated = np.zeros_like(x_values_lineal)

# Evalua cada polinomio de Lagrange en su intervalo 
for i, poly in enumerate(polys):
    mask = (x_values_lineal >= x_data_lineal[i]) & (x_values_lineal <= x_data_lineal[i+1])
    y_values_interpolated[mask] = poly(x_values_lineal[mask])

# Interpolación cúbica

x_interp_splines = np.linspace(-4, 4, 15)  
y_interp_splines = f_original(x_interp_splines)

spline = CubicSpline(x_interp_splines, y_interp_splines)

x_value_splines = np.linspace(-4, 4, 1000)
y_value_splines = spline(x_value_splines)
    



#error-----------

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



error_text_chevbyshev = calculate_errors(y_original, y_interpolated_bychev) ## es lo mismo pero mejor cambiar a como estan las otras
error_text_lineal = calculate_errors(f_original(x_values_lineal), y_values_interpolated)
error_text_spline = calculate_errors(f_original(x_value_splines), y_value_splines)


fig = plt.figure(figsize=(25, 6))



##graf lineal
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x_values_lineal, f_original(x_values_lineal), label='Función Original', color='orange')
ax1.plot(x_values_lineal, y_values_interpolated, label='Interpolación', linestyle='--', color='red')
ax1.scatter(x_data_lineal, y_data_lineal, label='Puntos de Datos', color='black')
ax1.set_title('Interpolación Lineal de Lagrange')
ax1.text(-4, np.min(f_original(x_values_lineal)), error_text_lineal, fontsize=8, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

##graf chebyshev
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x_chev_tograf, y_original, label='Función Original')
ax2.plot(x_chev_tograf, y_interpolated_bychev, label='Interpolación', linestyle='--', color='red')
ax2.plot(x_cheb, y_cheb, 'ro', label='Puntos de Chebyshev', color='black')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Interpolación con puntos de Chebyshev')
ax2.text(a, np.min(y_original), error_text_chevbyshev, fontsize=8, va='bottom', ha='left', color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax2.legend()
ax2.grid(True)

#falta poner aca la interpolacion polinomica



##graf spline
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x_value_splines, f_original(x_value_splines), label='Función Original', color='green')
ax4.plot(x_value_splines, y_value_splines, label='Interpolación', linestyle='--', color='red')
ax4.scatter(x_interp_splines, y_interp_splines, label='Puntos de Datos', color='black')
ax4.set_title('Interpolación con Splines Cubicos')
ax4.text(-4, np.min(f_original(x_value_splines)), error_text_spline, fontsize=8, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.legend()
ax4.grid(True)





plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()

