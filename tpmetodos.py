import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

def f_a(x):
    return (0.3**(abs(x))) * np.sin(4*x) - np.tanh(2*x) + 2

def f_b(x1, x2):
    return 0

def equispaced_dataset_build(start, stop, points_number, f): # points_number>=2
    data_set = []
    for x_i in np.linspace(start, stop, points_number):     # genera puntos equiespaciados
        data_set.append([x_i, f(x_i)])
    return data_set

def lagrange_interpolation(data_set):
    x_values = [point[0] for point in data_set]
    y_values = [point[1] for point in data_set]
    interpolator = lagrange(x_values, y_values)     
    return interpolator                             

def lagrange_iterator(start, stop, f, degree):
    degree_evolution = {}
    for i in range(2,degree+2):                                 # quiero generar hasta n+1 puntos, que me dará el mayor grado que quiero (n)
        data_set = equispaced_dataset_build(start, stop, i, f)
        degree_evolution[i-1]=lagrange_interpolation(data_set)
    return degree_evolution

def error_median(interpolator, f, x_values):
    y_values_interp = interpolator(x_values)
    y_values_exact = f(x_values)
    error = np.abs(y_values_exact - y_values_interp)
    return np.median(error)

def main():
    start = -4
    stop = 4
    degree = 30
    x_values = np.linspace(start, stop, 1000)
    errors = []
    interpolators_dict = lagrange_iterator(start, stop, f_a, degree)
    
    for i in range(2, degree+2):
        interpolator = interpolators_dict[i-1]
        error = error_median(interpolator, f_a, x_values)
        errors.append(error)
    
    plt.plot(range(1, degree + 1), errors, marker='o')
    plt.xlabel('Grado del polinomio interpolante')
    plt.ylabel('Error promedio de interpolación')
    plt.title('Error de interpolación vs. Grado del polinomio')
    plt.grid(True)
    plt.show()

    # podemos evaluar cual es el grado del polinomio que se corresponde con el menor error, y comparar ambos gráficos
        
    # y_values = f_a(x_values)

    # # Graficar la función
    # plt.plot(x_values, y_values, label='f_a(x)')
    # plt.xlabel('x')
    # plt.ylabel('f_a(x)')
    # plt.title('Gráfico de la función f_a(x)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()



