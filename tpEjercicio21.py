import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.interpolate import lagrange

# Leer los datos
measurements_df = pd.read_csv('mnyo_mediciones.csv', header=None, delim_whitespace=True)
ground_truth_df = pd.read_csv('mnyo_ground_truth.csv', header=None, delim_whitespace=True)

# Asignar nombres a las columnas
measurements_df.columns = ['x1', 'x2']
ground_truth_df.columns = ['x1', 'x2']

# Obtener los datos de las mediciones y el tiempo
x1_measurements = measurements_df['x1'].values
x2_measurements = measurements_df['x2'].values
time_measurements = np.arange(len(measurements_df))
time_interpolation = np.linspace(time_measurements.min(), time_measurements.max(), len(ground_truth_df))

# Crear interpolaciones lineales
interpolator_x1_linear = interp1d(time_measurements, x1_measurements, kind='linear')
interpolator_x2_linear = interp1d(time_measurements, x2_measurements, kind='linear')

# Crear interpolaciones cuadráticas
interpolator_x1_quadratic = interp1d(time_measurements, x1_measurements, kind='quadratic')
interpolator_x2_quadratic = interp1d(time_measurements, x2_measurements, kind='quadratic')

# Crear interpolación de Lagrange (de grado n-1, siendo n el número de datos)
interpolator_x1_lagrange = lagrange(time_measurements, x1_measurements)
interpolator_x2_lagrange = lagrange(time_measurements, x2_measurements)

# Crear interpolaciones cúbicas
tck_x1_cubic = splrep(time_measurements, x1_measurements, k=3)  # k=3 para un spline cúbico
tck_x2_cubic = splrep(time_measurements, x2_measurements, k=3)

# Calcular las posiciones interpoladas lineales
x1_interpolated_linear = interpolator_x1_linear(time_interpolation)
x2_interpolated_linear = interpolator_x2_linear(time_interpolation)

def interpolator_quadratic():
    return interpolator_x1_quadratic, interpolator_x2_quadratic

# Calcular las posiciones interpoladas cuadráticas
x1_interpolated_quadratic = interpolator_x1_quadratic(time_interpolation)
x2_interpolated_quadratic = interpolator_x2_quadratic(time_interpolation)

# Calcular las posiciones interpoladas de Lagrnage
x1_interpolated_lagrange = interpolator_x1_lagrange(time_interpolation)
x2_interpolated_lagrange = interpolator_x2_lagrange(time_interpolation)

# Calcular las posiciones interpoladas con spline cúbico
x1_interpolated_cubic = splev(time_interpolation, tck_x1_cubic)
x2_interpolated_cubic = splev(time_interpolation, tck_x2_cubic)

# Calcular los errores de cada interpolación
error_linear = np.sqrt((x1_interpolated_linear - ground_truth_df['x1'])**2 + (x2_interpolated_linear - ground_truth_df['x2'])**2)
error_quadratic = np.sqrt((x1_interpolated_quadratic - ground_truth_df['x1'])**2 + (x2_interpolated_quadratic - ground_truth_df['x2'])**2)
error_cubic = np.sqrt((x1_interpolated_cubic - ground_truth_df['x1'])**2 + (x2_interpolated_cubic - ground_truth_df['x2'])**2)
error_lagrange = np.sqrt((x1_interpolated_lagrange - ground_truth_df['x1'])**2 + (x2_interpolated_lagrange - ground_truth_df['x2'])**2)

# Calcular la mediana del error en cada caso
median_error_linear = np.median(error_linear)
median_error_quadratic = np.median(error_quadratic)
median_error_lagrange = np.median(error_lagrange)
median_error_cubic = np.median(error_cubic)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico para interpolación lineal
axes[0, 0].plot(x1_interpolated_linear, x2_interpolated_linear, label='Interpolación Lineal', color='green')
axes[0, 0].plot(ground_truth_df['x1'], ground_truth_df['x2'], label='Trayectoria Real', color='red', linestyle='--')
axes[0, 0].scatter(x1_measurements, x2_measurements, label='Puntos de Datos', color='black', s=50)
axes[0, 0].set_xlabel('x1')
axes[0, 0].set_ylabel('x2')
axes[0, 0].set_title(f'Interpolación Lineal (Error Mediano: {median_error_linear:.2f})')
axes[0, 0].legend()

# Gráfico para interpolación cuadrática
axes[0, 1].plot(x1_interpolated_quadratic, x2_interpolated_quadratic, label='Interpolación Cuadrática', color='blue')
axes[0, 1].plot(ground_truth_df['x1'], ground_truth_df['x2'], label='Trayectoria Real', color='red', linestyle='--')
axes[0, 1].scatter(x1_measurements, x2_measurements, label='Puntos de Datos', color='black', s=50)
axes[0, 1].set_xlabel('x1')
axes[0, 1].set_ylabel('x2')
axes[0, 1].set_title(f'Interpolación Cuadrática (Error Mediano: {median_error_quadratic:.2f})')
axes[0, 1].legend()

# Gráfico para interpolación Lagrange
axes[1, 0].plot(x1_interpolated_lagrange, x2_interpolated_lagrange, label='Interpolación Lagrange grado 9', color='pink')
axes[1, 0].plot(ground_truth_df['x1'], ground_truth_df['x2'], label='Trayectoria Real', color='red', linestyle='--')
axes[1, 0].scatter(x1_measurements, x2_measurements, label='Puntos de Datos', color='black', s=50)
axes[1, 0].set_xlabel('x1')
axes[1, 0].set_ylabel('x2')
axes[1, 0].set_title(f'Interpolación Lagrange grado 9 (Error Mediano: {median_error_lagrange:.2f})')
axes[1, 0].legend()

# Gráfico para interpolación cúbica
axes[1, 1].plot(x1_interpolated_cubic, x2_interpolated_cubic, label='Interpolación Cúbica', color='orange')
axes[1, 1].plot(ground_truth_df['x1'], ground_truth_df['x2'], label='Trayectoria Real', color='red', linestyle='--')
axes[1, 1].scatter(x1_measurements, x2_measurements, label='Puntos de Datos', color='black', s=50)
axes[1, 1].set_xlabel('x1')
axes[1, 1].set_ylabel('x2')
axes[1, 1].set_title(f'Interpolación Cúbica (Error Mediano: {median_error_cubic:.2f})')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# # Visualizar la trayectoria interpolada y la trayectoria real
# plt.figure(figsize=(10, 6))

# plt.plot(x1_interpolated_linear, x2_interpolated_linear, label=f'Interpolación Lineal (Error Mediano: {median_error_linear:.2f})', color='green')
# plt.plot(x1_interpolated_quadratic, x2_interpolated_quadratic, label=f'Interpolación Cuadrática (Error Mediano: {median_error_quadratic:.2f})', color='blue')
# plt.plot(x1_interpolated_lagrange, x2_interpolated_lagrange, label=f'Interpolación Lagrange grado 9 (Error Mediano: {median_error_lagrange:.2f})', color='pink')
# plt.plot(x1_interpolated_cubic, x2_interpolated_cubic, label=f'Interpolación Cúbica (Error Mediano: {median_error_cubic:.2f})', color='orange')
# plt.plot(ground_truth_df['x1'], ground_truth_df['x2'], label='Trayectoria Real', color='red', linestyle='--')
# plt.scatter(x1_measurements, x2_measurements, label='Puntos de Datos', color='black', s=50)  # Agregar puntos de datos
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Comparación de Interpolaciones con Trayectoria Real')
# plt.legend()
# plt.grid(True)
# plt.show()