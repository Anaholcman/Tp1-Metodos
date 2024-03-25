import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.interpolate import lagrange
from scipy.optimize import newton_krylov

# Leer los datos
measurements_t1 = pd.read_csv('mnyo_mediciones.csv', header=None, delim_whitespace=True)
measurements_t2 = pd.read_csv('mnyo_mediciones2.csv', header=None, sep='\s+')
ground_truth_t1 = pd.read_csv('mnyo_ground_truth.csv', header=None, sep='\s+')

# Asignar nombres a las columnas
measurements_t1.columns = ['x1', 'x2']
measurements_t2.columns = ['x1', 'x2']

# Obtener los datos de las mediciones y el tiempo
x1_measurements_t1 = measurements_t1['x1'].values
x2_measurements_t1 = measurements_t1['x2'].values
time_measurements_t1 = np.arange(len(measurements_t1))
# time_interpolation_t1 = np.linspace(time_measurements_t1.min(), time_measurements_t1.max(), len(ground_truth_df))

x1_measurements_t2 = measurements_t2['x1'].values
x2_measurements_t2 = measurements_t2['x2'].values
time_measurements_t2 = np.arange(len(measurements_t2))
# time_interpolation_t2 = np.linspace(time_measurements_t2.min(), time_measurements_t2.max(), len(ground_truth_t1))

# Crear interpolaciones cuadráticas
interpolator_t1_x1 = interp1d(time_measurements_t1, x1_measurements_t1, kind='quadratic')
interpolator_t1_x2 = interp1d(time_measurements_t1, x2_measurements_t1, kind='quadratic')

interpolator_t2_x1 = interp1d(time_measurements_t2, x1_measurements_t2, kind='quadratic')
interpolator_t2_x2 = interp1d(time_measurements_t2, x2_measurements_t2, kind='quadratic')

# test
def f(x):
    t = x[0]
    t_prime = x[1]
    x1 = interpolator_t1_x1(t) - interpolator_t2_x1(t_prime)
    x2 = interpolator_t1_x2(t) - interpolator_t2_x2(t_prime)
    return np.array([x1, x2])

# Definir la función que resuelve el sistema de ecuaciones no lineales con el método de Newton-Raphson
def find_intersection():
    initial_guess = [0.5, 0.5]  # Estimación inicial
    t_solution, t_prime_solution = newton_krylov(f, initial_guess, verbose=True)
    return t_solution, t_prime_solution

# Encontrar la intersección de las curvas
t_solution, t_prime_solution = find_intersection()

# Calcular las coordenadas de la intersección
x1_intersection_t1 = interpolator_t1_x1(t_solution)
x2_intersection_t1 = interpolator_t1_x2(t_solution)
x1_intersection_t2 = interpolator_t2_x1(t_prime_solution)
x2_intersection_t2 = interpolator_t2_x2(t_prime_solution)

# Imprimir los resultados
print("CONFIRMEMOS SI ES RAÍZ")
print(f([t_solution, t_prime_solution]))
print("Coordenadas de la intersección:")
print("Vehículo1_x1:", x1_intersection_t1)
print("Vehículo1_x2:", x2_intersection_t1)
print("Vehículo2_x1:", x1_intersection_t2)
print("Vehículo2_x2:", x2_intersection_t2)
print("t:", t_solution)
print("t_prime:", t_prime_solution)