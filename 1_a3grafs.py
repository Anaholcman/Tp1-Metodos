import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline



def f_original(x):
    return (0.3) ** np.abs(x) * np.sin(4 * x) - np.tanh(2 * x) + 2




##interpolacion lineal de lagrange----------------------

x_data_lineal = np.linspace(-4, 4, num=8)  
y_data_lineal = f_original(x_data_lineal)

polys = []

for i in range(len(x_data_lineal) - 1):
    poly = lagrange([x_data_lineal[i], x_data_lineal[i+1]], [y_data_lineal[i], y_data_lineal[i+1]])
    polys.append(poly)

x_values_lineal = np.linspace(-4, 4, 1000)# conjunto de puntos para graficar las interpolaciones lineales
y_values_interpolated = np.zeros_like(x_values_lineal)


for i, poly in enumerate(polys):# Evalua cada polinomio de Lagrange en su intervalo 
    mask = (x_values_lineal >= x_data_lineal[i]) & (x_values_lineal <= x_data_lineal[i+1])
    y_values_interpolated[mask] = poly(x_values_lineal[mask])


    
#interpolacion polinomica----------------

x_interp_lagrange = np.linspace(-4, 4, 8)
y_interp_lagrange = f_original(x_interp_lagrange)

interpolator_poli = lagrange(x_interp_lagrange, y_interp_lagrange)


x_values_polinomic = np.linspace(-4, 4, 1000)# Puntos para graficar
y_interp_points = interpolator_poli(x_values_polinomic)

# Interpolación cúbica----------

x_interp_splines = np.linspace(-4, 4, 8)  
y_interp_splines = f_original(x_interp_splines)

spline = CubicSpline(x_interp_splines, y_interp_splines)

x_value_splines = np.linspace(-4, 4, 1000)
y_splines_interpolated = spline(x_value_splines)

#interpolacion lagrange con chevbyshev--------
def chebyshev_points(a, b, n):
    k = np.arange(1, n+1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) / (2 * n) * np.pi)

a, b = -4, 4 
n = 8
x_cheb = chebyshev_points(a, b, n)
y_cheb = f_original(x_cheb)

poly = lagrange(x_cheb, y_cheb) 


x_chev_tograf = np.linspace(a, b, 1000)# Puntos para graficar la función original
y_original = f_original(x_chev_tograf)
y_interpolated_bychev = poly(x_chev_tograf) 

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
error_text_spline = calculate_errors(f_original(x_value_splines), y_splines_interpolated)
error_text_polinomic = calculate_errors(f_original(x_values_polinomic), y_interp_points)


fig, ax = plt.subplots(3, 1, figsize=(8, 12))

# Interpolación lineal
ax[0].plot(x_values_lineal, f_original(x_values_lineal), label='Función Original', color='blue')
ax[0].plot(x_values_lineal, y_values_interpolated, label='Interpolación', linestyle='--', color='red')
ax[0].scatter(x_data_lineal, y_data_lineal, label='Puntos de Datos', color='black')
ax[0].set_title('Interpolación Lineal')
ax[0].text(-4, np.min(f_original(x_values_lineal)), error_text_lineal, fontsize=8, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].legend()
ax[0].grid(True)

# Interpolación polinómica de Lagrange
ax[1].plot(x_values_polinomic, f_original(x_values_polinomic), label='Función Original', color='orange')
ax[1].plot(x_values_polinomic, y_interp_points, label='Interpolación', linestyle='--', color='red')
ax[1].scatter(x_interp_lagrange, y_interp_lagrange, label='Puntos de Datos', color='black')
ax[1].set_title('Interpolación Polinómica de Lagrange')
ax[1].text(-4, np.min(f_original(x_values_polinomic)), error_text_polinomic, fontsize=8, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].legend()
ax[1].grid(True)

# Graficar Splines
ax[2].plot(x_value_splines, f_original(x_value_splines), label='Función Original', color='green')
ax[2].plot(x_value_splines, y_splines_interpolated, label='Interpolación', linestyle='--', color='red')
ax[2].scatter(x_interp_splines, y_interp_splines, label='Puntos de Datos', color='black')
ax[2].set_title('Interpolación con Splines Cúbicos')
ax[2].text(-4, np.min(f_original(x_value_splines)), error_text_spline, fontsize=8, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
ax[2].legend()
ax[2].grid(True)





plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()

##generador de distintos puntos 
x_interp_lagrange_10 = np.linspace(-4, 4, 10)
y_interp_lagrange_10 = f_original(x_interp_lagrange_10)
interpolator_poli_10 = lagrange(x_interp_lagrange_10, y_interp_lagrange_10)
error_text_lagrange10 = calculate_errors(f_original(x_values_polinomic), interpolator_poli_10(x_values_polinomic))

x_interp_lagrange_15 = np.linspace(-4, 4, 15)
y_interp_lagrange_15 = f_original(x_interp_lagrange_15)
interpolator_poli_15 = lagrange(x_interp_lagrange_15, y_interp_lagrange_15)
error_text_lagrange15 = calculate_errors(f_original(x_values_polinomic), interpolator_poli_15(x_values_polinomic))

x_cheb10 = chebyshev_points(a, b, 10)
y_cheb10 = f_original(x_cheb10)
poly10 = lagrange(x_cheb10, y_cheb10)

y_interpolated_bychev10 = poly10(x_chev_tograf)
error_text_chevbyshev10 = calculate_errors(y_original, y_interpolated_bychev10)




fig2 = plt.figure(figsize=(10, 7))

ax1 = fig2.add_subplot(3, 2, 1)
ax1.plot(x_values_polinomic, f_original(x_values_polinomic), label='Función Original', color='orange')
ax1.plot(x_values_polinomic, y_interp_points, label='Interpolación', linestyle='--', color='red')
ax1.scatter(x_interp_lagrange, y_interp_lagrange, label='Puntos de Datos', color='black')
ax1.set_title('Lagrange con puntos equiespaciados')
ax1.text(-4, np.min(f_original(x_values_polinomic)), error_text_polinomic, fontsize=6, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

##graf chebyshev
ax2 = fig2.add_subplot(3, 2, 2)
ax2.plot(x_chev_tograf, y_original, label='Función Original')
ax2.plot(x_chev_tograf, y_interpolated_bychev, label='Interpolación', linestyle='--', color='red')
ax2.plot(x_cheb, y_cheb, label='Puntos de Chebyshev', color='black')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Lagrange con puntos de Chebyshev')
ax2.text(-4, np.min(y_original), error_text_chevbyshev, fontsize=6, va='bottom', ha='left', color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax2.legend()
ax2.grid(True)

##lagrange 10 puntos
ax3 = fig2.add_subplot(3, 2, 3)
ax3.plot(x_values_polinomic, f_original(x_values_polinomic), label='Función Original', color='orange')
ax3.plot(x_values_polinomic, interpolator_poli_10(x_values_polinomic), label='Interpolación', linestyle='--', color='red')
ax3.scatter(x_interp_lagrange_10, y_interp_lagrange_10, label='Puntos de Datos', color='black')
ax3.set_title('10 puntos equiespaciados')
ax3.text(-4, np.min(f_original(x_values_polinomic))-2.3, error_text_lagrange10, fontsize=6, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.legend()
ax3.grid(True)

ax4 = fig2.add_subplot(3, 2, 4)
ax4.plot(x_chev_tograf, y_original, label='Función Original')
ax4.plot(x_chev_tograf, y_interpolated_bychev10, label='Interpolación', linestyle='--', color='red')
ax4.plot(x_cheb10, y_cheb10, label='Puntos de Chebyshev', color='black')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Lagrange con 10 puntos de Chebyshev')
ax4.text(-4, np.min(y_original), error_text_chevbyshev10, fontsize=6, va='bottom', ha='left', color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax4.legend()
ax4.grid(True)



##lagrange 15 puntos
ax5 = fig2.add_subplot(3, 2, 5)
ax5.plot(x_values_polinomic, f_original(x_values_polinomic), label='Función Original', color='orange')
ax5.plot(x_values_polinomic, interpolator_poli_15(x_values_polinomic), label='Interpolación', linestyle='--', color='red')
ax5.scatter(x_interp_lagrange_15, y_interp_lagrange_15, label='Puntos de Datos', color='black')
ax5.set_title('15 puntos equiespaciados')
ax5.text(-4, np.min(f_original(x_values_polinomic))-2.5, error_text_lagrange15, fontsize=6, va='bottom', ha='left', color='red',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.legend()
ax5.grid(True)







plt.tight_layout()
plt.show()

