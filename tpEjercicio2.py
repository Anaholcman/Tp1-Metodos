import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev

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

# Crear interpolaciones cúbicas
tck_x1_cubic = splrep(time_measurements, x1_measurements, k=3)  # k=3 para un spline cúbico
tck_x2_cubic = splrep(time_measurements, x2_measurements, k=3)

# Calcular las posiciones interpoladas con spline cúbico
x1_interpolated_cubic = splev(time_interpolation, tck_x1_cubic)
x2_interpolated_cubic = splev(time_interpolation, tck_x2_cubic)

# Calcular las posiciones interpoladas lineales
x1_interpolated_linear = interpolator_x1_linear(time_interpolation)
x2_interpolated_linear = interpolator_x2_linear(time_interpolation)

# Calcular las posiciones interpoladas cuadráticas
x1_interpolated_quadratic = interpolator_x1_quadratic(time_interpolation)
x2_interpolated_quadratic = interpolator_x2_quadratic(time_interpolation)

# Calcular los errores de cada interpolación
error_linear = np.sqrt((x1_interpolated_linear - ground_truth_df['x1'])**2 + (x2_interpolated_linear - ground_truth_df['x2'])**2)
error_quadratic = np.sqrt((x1_interpolated_quadratic - ground_truth_df['x1'])**2 + (x2_interpolated_quadratic - ground_truth_df['x2'])**2)
error_cubic = np.sqrt((x1_interpolated_cubic - ground_truth_df['x1'])**2 + (x2_interpolated_cubic - ground_truth_df['x2'])**2)

# Calcular la mediana del error en cada caso
median_error_linear = np.median(error_linear)
median_error_quadratic = np.median(error_quadratic)
median_error_cubic = np.median(error_cubic)

# Visualizar la trayectoria interpolada y la trayectoria real
plt.figure(figsize=(10, 6))

plt.plot(x1_interpolated_linear, x2_interpolated_linear, label=f'Interpolación Lineal (Error Mediano: {median_error_linear:.2f})', color='green')
plt.plot(x1_interpolated_quadratic, x2_interpolated_quadratic, label=f'Interpolación Cuadrática (Error Mediano: {median_error_quadratic:.2f})', color='blue')
plt.plot(x1_interpolated_cubic, x2_interpolated_cubic, label=f'Interpolación Cúbica (Error Mediano: {median_error_cubic:.2f})', color='orange')
plt.plot(ground_truth_df['x1'], ground_truth_df['x2'], label='Trayectoria Real', color='red', linestyle='--')
plt.scatter(x1_measurements, x2_measurements, label='Puntos de Datos', color='black', s=50)  # Agregar puntos de datos
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Comparación de Interpolaciones con Trayectoria Real')
plt.legend()
plt.grid(True)
plt.show()