import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, interp1d, CubicSpline
from scipy.interpolate import splrep, splev

def f_a(x):
    return (0.3**(abs(x))) * np.sin(4*x) - np.tanh(2*x) + 2

def equispaced_dataset_build(start, stop, points_number, f):
    data_set = []
    for x_i in np.linspace(start, stop, points_number):
        data_set.append([x_i, f(x_i)])
    return data_set

def lagrange_interpolation(data_set):
    x_values = [point[0] for point in data_set]
    y_values = [point[1] for point in data_set]
    interpolator = lagrange(x_values, y_values)
    return interpolator

def cubic_spline_interpolation(data_set):
    x_values = [point[0] for point in data_set]
    y_values = [point[1] for point in data_set]
    interpolator = CubicSpline(x_values, y_values)
    return interpolator

def linear_interpolation(data_set):
    x_values = [point[0] for point in data_set]
    y_values = [point[1] for point in data_set]
    interpolator = interp1d(x_values, y_values, kind='linear', bounds_error=False)
    return interpolator

def quadratic_interpolation(data_set):
    x_values = [point[0] for point in data_set]
    y_values = [point[1] for point in data_set]
    if len(x_values) >= 3:
        interpolator = splrep(x_values, y_values, k=2)
    else:
        interpolator = interp1d(x_values, y_values, kind='linear', bounds_error=False)
    return interpolator


def error_median(interpolator, f, x_values):
    if isinstance(interpolator, tuple):
        try:
            y_values_interp = splev(x_values, interpolator)
        except ValueError:
            # En caso de que haya problemas con la evaluación del spline, se devuelve None
            y_values_interp = None
    else:
        y_values_interp = interpolator(x_values)
    if y_values_interp is None:
        return np.inf  # Devolver un error infinito en caso de problemas con la evaluación del spline
    else:
        y_values_exact = f(x_values)
        error = np.abs(y_values_exact - y_values_interp)
        return np.median(error)

def main():
    start = -4
    stop = 4
    degree = 30
    x_values = np.linspace(start, stop, 1000)
    errors = []

    interpolators_dict = {
        'Lagrange': lagrange_interpolation,
        'Quadratic': quadratic_interpolation,
        'Linear': linear_interpolation,
        'Cubic Spline': cubic_spline_interpolation
    }

    # Generar gráfico de error promedio vs. cantidad de puntos de interpolación
    plt.figure(figsize=(10, 6))
    for interp_name, interp_func in interpolators_dict.items():
        errors_list = []
        for i in range(2, degree + 2):
            data_set = equispaced_dataset_build(start, stop, i, f_a)
            interpolator = interp_func(data_set)
            error = error_median(interpolator, f_a, x_values)
            errors_list.append(error)
        errors.append(errors_list)
        plt.plot(range(2, degree + 2), errors_list, marker='o', label=interp_name)

    plt.xlabel('Cantidad de puntos de interpolación')
    plt.ylabel('Error promedio de interpolación')
    plt.title('Error de interpolación vs. Cantidad de puntos de interpolación')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Graficar la función original y sus interpolaciones en un gráfico separado
    plt.figure(figsize=(10, 6))

    for interp_name, interp_func in interpolators_dict.items():
        data_set = equispaced_dataset_build(start, stop, 10, f_a)
        interpolator = interp_func(data_set)
        yp_values = interpolator(x_values)
        if interp_name == 'Lagrange':
            plt.plot(x_values, yp_values, label=interp_name, linestyle='--', color='blue')
        elif interp_name == 'Quadratic':
            plt.plot(x_values, yp_values, label=interp_name, linestyle='--', color='green')
        elif interp_name == 'Linear':
            plt.plot(x_values, yp_values, label=interp_name, linestyle='--', color='orange')
        elif interp_name == 'Cubic Spline':
            plt.plot(x_values, yp_values, label=interp_name, linestyle='--', color='red')

    # Agregar la función original en negro
    y_values = f_a(x_values)
    plt.plot(x_values, y_values, label='f_a(x)', color='black')

    plt.xlabel('x')
    plt.ylabel('f_a(x)')
    plt.title('Gráfico de la función f_a(x) y su interpolación')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
