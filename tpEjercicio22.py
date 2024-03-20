import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.interpolate import lagrange
import tpEjercicio21 as ej2
from scipy.optimize import newton_krylov

# Leer los datos
measurements_df = pd.read_csv('mnyo_mediciones2.csv', header=None, delim_whitespace=True)
ground_truth_df = pd.read_csv('mnyo_ground_truth.csv', header=None, delim_whitespace=True)

# Asignar nombres a las columnas
measurements_df.columns = ['x1', 'x2']

# Obtener los datos de las mediciones y el tiempo
x1_measurements = measurements_df['x1'].values
x2_measurements = measurements_df['x2'].values
time_measurements = np.arange(len(measurements_df))
time_interpolation = np.linspace(time_measurements.min(), time_measurements.max(), len(ground_truth_df))

# Crear interpolaciones cuadráticas
interpolator2_x1_quadratic = interp1d(time_measurements, x1_measurements, kind='quadratic')
interpolator2_x2_quadratic = interp1d(time_measurements, x2_measurements, kind='quadratic')

#
interpolator1_x1_quadratic = ej2.interpolator_x1_quadratic
interpolator_x2_quadratic = ej2.interpolator_x2_quadratic

# # Definir la función que representa la diferencia entre las interpolaciones
# def difference_function(t, k):
#     x1_i1 = interpolator2_x1_quadratic(t)
#     x2_i1 = interpolator2_x2_quadratic(t)
#     x1_i2 = interpolator1_x1_quadratic(k)
#     x2_i2 = interpolator_x2_quadratic(k)
#     return np.array([x1_i1 - x1_i2, x2_i1 - x2_i2])

# # Adivinanza inicial para el método de Newton
# t_initial_guess = 0.5
# k_initial_guess = 0.5

# # Resolver el sistema de ecuaciones no lineales con el método de Newton multivariado
# t_solution, k_solution = newton_krylov(difference_function, [t_initial_guess, k_initial_guess])

# # Calcular las coordenadas de la intersección
# x1_intersection = interpolator2_x1_quadratic(t_solution)
# x2_intersection = interpolator2_x2_quadratic(t_solution)

# print("Coordenadas de la intersección:")
# print("x1:", x1_intersection)
# print("x2:", x2_intersection)
# print("t:", t_solution)
# print("k:", k_solution)